#!/usr/bin/env python3
"""Run the RegIntel GPU index build on Kaggle, then verify it locally.

    python scripts/kaggle_build_index.py                # stage, push, wait, fetch, verify
    python scripts/kaggle_build_index.py --skip-dataset # dataset already current
    python scripts/kaggle_build_index.py --skip-push    # fetch the last outputs only
    python scripts/kaggle_build_index.py --verify-only  # compare fetched vs local only

Steps: stage the code (regintel/) + corpus (data/chunks.jsonl, documents.jsonl)
into the private dataset <user>/regintel-src (created or a new version); wait
until the dataset is visible to the API; push kaggle/gpu_index_build/kernel.py
as GPU kernel <user>/regintel-gpu-index-build; wait for the run; download the
outputs (index artifacts + run report) to kaggle/out/; verify the GPU-built
vectors against the local CPU build (cosine equivalence).

Hard-won details this encodes (kaggle CLI 2.x):
  * `datasets create/version` SKIPS folders unless --dir-mode zip is given —
    a bare create uploads zero files, and the CLI still exits 0 while printing
    "Dataset creation error", so the output text is checked too;
  * re-uploads must go through `datasets version`: `create` fails with
    "title ... is already in use" (not "already exists"/409), so any create
    failure falls back to a version upload;
  * a kernel pushed with an invalid dataset source is accepted but the source
    is silently dropped ("could not be added"), so the push output is checked;
  * the slug of a brand-new kernel is derived from its title, so the real ref
    is resolved from the push output instead of trusting the requested slug;
  * a dataset that has not finished processing is not mounted into the kernel,
    so the script waits for API visibility before pushing — probed with
    `datasets files <ref>` because CLI 2.x dropped `datasets list --page-size`
    (list page size is fixed at 20, and a fresh private dataset may not be on
    the first page).

Kaggle credentials: ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

DATASET_SLUG = "regintel-src"
KERNEL_SLUG = "regintel-gpu-index-build"
OUT_DIR = ROOT / "kaggle" / "out"
_UPLOAD_ERROR = re.compile(r"dataset (creation|version) error", re.I)
_REF_RE = re.compile(r"kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)")


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    return [exe] if exe else [sys.executable, "-m", "kaggle"]


def username() -> str:
    user = os.environ.get("KAGGLE_USERNAME", "")
    if user:
        return user
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8"))["username"]
    raise SystemExit("no Kaggle credentials: put kaggle.json in ~/.kaggle "
                     "or set KAGGLE_USERNAME/KAGGLE_KEY")


def _run_capture(cmd: list[str]) -> tuple[int, str]:
    # PYTHONUTF8/PYTHONIOENCODING: on Windows the Kaggle CLI crashes with a
    # 'charmap' encode error (exit 1) when its output contains non-cp1252
    # characters, even after it has done its work.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def run(cmd: list[str]) -> str:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc, blob = _run_capture(cmd)
    if rc != 0:
        sys.stderr.write(blob + "\n")
        raise SystemExit(f"command failed ({rc}): {' '.join(cmd)}")
    if blob:
        print(blob[:4000], flush=True)
    return blob


def stage_dataset(user: str) -> Path:
    """Copy code + corpus into a temp dataset folder (never touches the repo)."""
    stage = Path(tempfile.mkdtemp(prefix="regintel_kaggle_")) / DATASET_SLUG
    stage.mkdir(parents=True)
    shutil.copytree(ROOT / "regintel", stage / "regintel",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (stage / "data").mkdir()
    for name in ("chunks.jsonl", "documents.jsonl"):
        shutil.copy(ROOT / "data" / name, stage / "data" / name)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "RegIntel source + corpus (private)",
        "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    print(f"staged {size / 1e6:.1f} MB -> {stage}", flush=True)
    return stage


def upsert_dataset(user: str, stage: Path) -> None:
    create = kaggle_cmd() + ["datasets", "create", "-p", str(stage), "--dir-mode", "zip"]
    rc, blob = _run_capture(create)
    if rc == 0 and not _UPLOAD_ERROR.search(blob):
        print(blob[:2000], flush=True)
        return
    # Re-uploads fail at `create` with "title ... is already in use" (the CLI
    # does not say "already exists" / 409). Any create failure falls through to
    # a new version of the existing dataset; if that fails too, surface both.
    version = kaggle_cmd() + ["datasets", "version", "-p", str(stage), "--dir-mode", "zip",
                              "-m", "refresh code + corpus"]
    rc2, blob2 = _run_capture(version)
    if rc2 == 0 and not _UPLOAD_ERROR.search(blob2):
        print(blob2[:2000], flush=True)
        return
    sys.stderr.write(blob + "\n" + blob2 + "\n")
    raise SystemExit("dataset upload failed (see the CLI output above)")


def wait_dataset_ready(user: str, timeout_min: int = 10) -> None:
    """A freshly uploaded dataset is not mountable until Kaggle processes it.

    Probe with `datasets files <ref>`: it fails while the dataset is still
    being created and lists rows once it is queryable. (CLI 2.x removed
    `datasets list --page-size`; the list is fixed at 20 rows/page.)"""
    ref = f"{user}/{DATASET_SLUG}"
    deadline = time.time() + timeout_min * 60
    while True:
        rc, blob = _run_capture(kaggle_cmd() + ["datasets", "files", ref])
        if rc == 0 and "name" in blob.lower() and "error" not in blob.lower():
            print(f"dataset {ref} is visible; letting Kaggle finish processing", flush=True)
            time.sleep(45)
            return
        if time.time() > deadline:
            raise SystemExit(f"dataset {ref} never became visible to the API")
        time.sleep(20)


def push_kernel(user: str) -> str:
    src = ROOT / "kaggle" / "gpu_index_build"
    stage = Path(tempfile.mkdtemp(prefix="regintel_kernel_"))
    shutil.copy(src / "kernel.py", stage / "kernel.py")
    meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = f"{user}/{KERNEL_SLUG}"
    meta["dataset_sources"] = [f"{user}/{DATASET_SLUG}"]
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    blob = run(kaggle_cmd() + ["kernels", "push", "-p", str(stage)])
    if "not valid dataset sources" in blob:
        raise SystemExit("kernel was pushed WITHOUT its dataset source — is the dataset processed?")
    m = _REF_RE.search(blob)
    return f"{m.group(1)}/{m.group(2)}" if m else f"{user}/{KERNEL_SLUG}"


def wait_kernel(ref: str, timeout_min: int = 60) -> str:
    deadline = time.time() + timeout_min * 60
    checks = 0
    while True:
        _rc, blob = _run_capture(kaggle_cmd() + ["kernels", "status", ref])
        checks += 1
        if "COMPLETE" in blob:
            print(blob[:300], flush=True)
            return "COMPLETE"
        if any(s in blob for s in ("ERROR", "CANCELLED", "FAILED")):
            print(blob[:300], flush=True)
            return "ERROR"
        if checks in (5, 20, 40):
            print(f"still running ({checks} checks): {blob[:200]}", flush=True)
        if time.time() > deadline:
            print("timed out waiting for the kernel", flush=True)
            return "TIMEOUT"
        time.sleep(30)


def fetch_outputs(ref: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run(kaggle_cmd() + ["kernels", "output", ref, "-p", str(OUT_DIR)])


def _find(name: str) -> Path | None:
    hits = sorted(OUT_DIR.rglob(name))
    return hits[0] if hits else None


def verify() -> dict:
    """Cosine equivalence between the GPU build and the local CPU build."""
    import numpy as np

    report: dict = {"fetched": sorted(str(p.relative_to(OUT_DIR)) for p in OUT_DIR.rglob("*")
                                      if p.is_file())}
    report_path = _find("gpu_report.json")
    if report_path:
        report["gpu_report"] = json.loads(report_path.read_text(encoding="utf-8"))
    local_p, remote_p = ROOT / "data" / "index" / "dense_fp16.npy", _find("dense_fp16.npy")
    if local_p.exists() and remote_p:
        a = np.load(local_p).astype(np.float32)
        b = np.load(remote_p).astype(np.float32)
        if a.shape == b.shape:
            a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
            b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            cos = (a * b).sum(axis=1)
            report.update({"n": int(a.shape[0]), "cos_mean": round(float(cos.mean()), 6),
                           "cos_min": round(float(cos.min()), 6),
                           "equivalent": bool(cos.min() > 0.999)})
        else:
            report["shape_mismatch"] = [list(a.shape), list(b.shape)]
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kaggle GPU index build for RegIntel")
    ap.add_argument("--skip-dataset", action="store_true", help="dataset already current")
    ap.add_argument("--skip-push", action="store_true", help="fetch the last outputs only")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--wait-min", type=int, default=60)
    args = ap.parse_args(argv)

    user = username()
    status = None
    ref = f"{user}/{KERNEL_SLUG}"
    if not args.verify_only:
        if not args.skip_push:
            if not args.skip_dataset:
                stage = stage_dataset(user)
                upsert_dataset(user, stage)
                wait_dataset_ready(user)
            ref = push_kernel(user)
            print(f"kernel ref: {ref}", flush=True)
            status = wait_kernel(ref, timeout_min=args.wait_min)
        fetch_outputs(ref)
    verify()
    return 0 if status in (None, "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
