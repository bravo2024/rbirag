"""Kaggle GPU kernel: rebuild the RegIntel retrieval index on a GPU.

Runs on Kaggle (pushed by scripts/kaggle_build_index.py), reads the code +
corpus from the private dataset `regintel-src`, encodes all chunks with the
same BAAI/bge-small-en-v1.5 model the local build uses (seconds on GPU vs
tens of minutes on CPU), rebuilds BM25 and writes the full index set —
dense_fp16.npy, dense_meta.json, bm25.npz, bm25_vocab.json, manifest.json —
to /kaggle/working/data. The local orchestrator downloads the outputs and
verifies them against the CPU build (cosine equivalence).

The run report (/kaggle/working/gpu_report.json) is written on success AND on
failure (with the traceback): the Kaggle CLI cannot stream a run log, so the
report is the diagnosis channel.
"""
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
DATA = WORK / "data"


def prepare_src() -> Path:
    """Return a directory containing regintel/ and data/chunks.jsonl.

    Handles both dataset layouts: folders uploaded directly, and zips left
    behind by `kaggle datasets ... --dir-mode zip` (extracted into
    /kaggle/working/src).
    """
    inputs = Path("/kaggle/input")
    candidates = [p for p in [inputs / "regintel-src", *sorted(inputs.glob("*"))] if p.is_dir()]
    for cand in candidates:
        hits = sorted(cand.rglob("chunks.jsonl"))
        if hits:
            return hits[0].parent.parent
    zips = [z for cand in candidates for z in cand.rglob("*.zip")]
    if zips:
        src = WORK / "src"
        src.mkdir(parents=True, exist_ok=True)
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(src)
        hits = sorted(src.rglob("chunks.jsonl"))
        if hits:
            return hits[0].parent.parent
    raise FileNotFoundError(
        f"regintel-src not mounted; /kaggle/input contains {[p.name for p in inputs.glob('*')]}")


def main() -> None:
    t0 = time.time()
    report: dict = {"status": "error"}
    try:
        src = prepare_src()
        report["src"] = str(src)
        DATA.mkdir(parents=True, exist_ok=True)
        for name in ("chunks.jsonl", "documents.jsonl"):
            shutil.copy(src / "data" / name, DATA / name)
        os.environ["REGINTEL_DATA_DIR"] = str(DATA)   # writable corpus location
        sys.path.insert(0, str(src))

        try:
            import faiss  # noqa: F401
        except ImportError:
            os.system(f"{sys.executable} -m pip install -q faiss-cpu")

        import torch
        import sentence_transformers

        from regintel import ingest

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        report.update({"gpu": gpu, "device": "cuda" if torch.cuda.is_available() else "cpu",
                       "torch": torch.__version__,
                       "sentence_transformers": sentence_transformers.__version__})
        print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpu={gpu} | "
              f"sentence-transformers {sentence_transformers.__version__}", flush=True)

        manifest = ingest.rebuild_indexes(
            progress=lambda *a: print(*a, flush=True), full_dense=True)
        report.update({"status": "ok", "n_chunks": manifest["n_chunks"],
                       "n_docs": manifest["n_docs"], "version": manifest["version"]})
        for name in ("chunks.jsonl", "documents.jsonl"):   # keep the outputs lean
            (DATA / name).unlink(missing_ok=True)
    except Exception:
        report["traceback"] = traceback.format_exc()
    finally:
        report["seconds"] = round(time.time() - t0, 1)
        (WORK / "gpu_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("DONE " + json.dumps(report), flush=True)
    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
