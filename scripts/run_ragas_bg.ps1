param(
    [string]$File = "evals/eval_set_v2.jsonl",
    [string[]]$Category = @("answerable", "amendment", "as_of"),
    [string[]]$JudgeModel = @(),
    [int]$ContextSize = 0,
    [string]$ContextScope = "total",
    [string]$Answers = "",
    [int]$Cores = 2,
    [int]$Limit = 0,
    [string]$Tag = "v2_ragas_full_draft",
    [switch]$Agreement,
    [switch]$DryRun,
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:RAGAS_DO_NOT_TRACK = "true"

$cores = [Math]::Max(1, [Math]::Min($Cores, [Environment]::ProcessorCount))
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $root "evals/results"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$log = Join-Path $outDir "ragas_${Tag}_${stamp}.log"
$err = "${log}.err"
$meta = "${log}.meta"
$mask = [IntPtr]([int64](1 -shl $cores) - 1)

$pyArgs = @("-u", "scripts/eval_answers.py")
if ($Answers) {
    $pyArgs += @("--answers", $Answers)
} else {
    $pyArgs += @("--file", $File)
    foreach ($c in $Category) { $pyArgs += @("--category", $c) }
    if ($Limit -gt 0) { $pyArgs += @("--limit", "$Limit") }
}
foreach ($jm in $JudgeModel) { $pyArgs += @("--judge-model", $jm) }
if ($ContextSize -gt 0) { $pyArgs += @("--context-size", "$ContextSize", "--context-scope", $ContextScope) }
if (-not $Agreement) { $pyArgs += "--skip-agreement" }
$pyArgs += @("--tag", $Tag)

if ($DryRun) {
    "python $($pyArgs -join ' ')"
    "cores=$cores affinity_mask=$mask log=$log"
    exit 0
}

if ($Detach) {
    $p = Start-Process -FilePath "python" -ArgumentList $pyArgs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err
} else {
    $p = Start-Process -FilePath "python" -ArgumentList $pyArgs -PassThru `
        -NoNewWindow -RedirectStandardOutput $log -RedirectStandardError $err
}

$priority = "unset"
$affinity = "unset"
try {
    $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    $priority = "BelowNormal"
} catch {
    $priority = "FAILED: $($_.Exception.Message)"
}
try {
    $p.ProcessorAffinity = $mask
    $affinity = "ok"
} catch {
    $affinity = "FAILED: $($_.Exception.Message)"
}

"pid=$($p.Id) tag=$Tag detach=$([bool]$Detach) file=$File answers=$Answers judge=$($JudgeModel -join ' -> ') context=$ContextSize/$ContextScope categories=$($Category -join ',') limit=$Limit agreement=$([bool]$Agreement) cores=$cores mask=$mask priority=$priority affinity=$affinity log=$log err=$err" |
    Tee-Object -FilePath $meta
if (-not $Detach) {
    $p.WaitForExit()
    "exit=$($p.ExitCode) finished=$(Get-Date -Format o)" | Tee-Object -Append -FilePath $meta
    exit $p.ExitCode
}

