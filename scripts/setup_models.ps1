param(
    [switch]$SkipSam,
    [switch]$SkipLivePortrait,
    [switch]$SkipDecaPrepare,
    [switch]$SkipRepos
)

$ErrorActionPreference = "Stop"

Write-Host "[*] FaceWarp Lab model setup started"

# Project root
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv\Scripts\python.exe"
if (!(Test-Path $Python)) {
    throw "Project venv Python not found at $Python. Create/restore venv before running setup_models.ps1."
}

# Make dirs
New-Item -ItemType Directory -Force local_models | Out-Null
New-Item -ItemType Directory -Force local_models\sam_aging | Out-Null
New-Item -ItemType Directory -Force local_models\liveportrait_weights | Out-Null
New-Item -ItemType Directory -Force local_models\deca_data\weights | Out-Null
New-Item -ItemType Directory -Force local_models\deca_data\FLAME2020 | Out-Null
New-Item -ItemType Directory -Force local_models\albedogan_weights | Out-Null

# Keep local_models directory in git, but not its downloaded contents.
New-Item -ItemType File -Force local_models\.gitkeep | Out-Null

# Clone external repos if missing.
# These repos are intentionally not committed to this project.
if (!$SkipRepos) {
    Write-Host "[*] Checking external model repositories"

    if (!(Test-Path "local_models\DECA")) {
        Write-Host "[*] Cloning DECA"
        git clone https://github.com/yfeng95/DECA.git local_models\DECA
    }
    else {
        Write-Host "[+] DECA repo already exists"
    }

    if (!(Test-Path "local_models\SAM")) {
        Write-Host "[*] Cloning SAM"
        git clone https://github.com/yuval-alaluf/SAM.git local_models\SAM
    }
    else {
        Write-Host "[+] SAM repo already exists"
    }

    if (!(Test-Path "local_models\LivePortrait")) {
        Write-Host "[*] Cloning LivePortrait"
        git clone https://github.com/KwaiVGI/LivePortrait.git local_models\LivePortrait
    }
    else {
        Write-Host "[+] LivePortrait repo already exists"
    }

    if (!(Test-Path "local_models\face-parsing.PyTorch")) {
        Write-Host "[*] Cloning face-parsing.PyTorch"
        git clone https://github.com/zllrunning/face-parsing.PyTorch.git local_models\face-parsing.PyTorch
    }
    else {
        Write-Host "[+] face-parsing.PyTorch repo already exists"
    }
}
else {
    Write-Host "[*] Skipping external repo clone"
}

# Ensure Hugging Face CLI
Write-Host "[*] Installing / checking Hugging Face CLI"
& $Python -m pip install -U huggingface_hub

$hfCmd = "hf"

try {
    & $hfCmd --help | Out-Null
}
catch {
    $hfCmd = ".\venv\Scripts\hf.exe"

    if (!(Test-Path $hfCmd)) {
        throw "hf command not found. Activate venv first, then run this script."
    }
}

# SAM Aging
if (!$SkipSam) {
    Write-Host "[*] Downloading SAM aging weight"

    try {
        & $hfCmd download aimi-models/sam-aging sam_ffhq_aging.pt --local-dir local_models\sam_aging
    }
    catch {
        Write-Host "[!] SAM weight download failed: $($_.Exception.Message)"
        Write-Host "    Continuing because SAM is an optional aging model slot."
    }

    if (Test-Path "local_models\sam_aging\sam_ffhq_aging.pt") {
        Write-Host "[+] SAM weight ready: local_models\sam_aging\sam_ffhq_aging.pt"
    }
    else {
        Write-Host "[-] SAM weight missing after download"
    }
}
else {
    Write-Host "[*] Skipping SAM download"
}

# LivePortrait
if (!$SkipLivePortrait) {
    Write-Host "[*] Downloading LivePortrait weights"

    try {
        & $hfCmd download KlingTeam/LivePortrait --local-dir local_models\liveportrait_weights
    }
    catch {
        Write-Host "[!] LivePortrait weight download failed: $($_.Exception.Message)"
        Write-Host "    Continuing because LivePortrait is an optional expression model slot."
    }

    Write-Host "[*] Mirroring LivePortrait weights into repo expected folder"

    $srcRoot = "local_models\liveportrait_weights"
    $targetRoot = "local_models\LivePortrait\pretrained_weights"
    $nested = Join-Path $srcRoot "pretrained_weights"

    if (Test-Path $nested) {
        $copySource = $nested
    }
    else {
        $copySource = $srcRoot
    }

    New-Item -ItemType Directory -Force $targetRoot | Out-Null

    if (Test-Path $copySource) {
        Get-ChildItem -Path $copySource -Force |
            Where-Object { $_.Name -ne ".cache" } |
            ForEach-Object {
                Copy-Item $_.FullName $targetRoot -Recurse -Force
            }

        Write-Host "[+] LivePortrait weights mirrored to: $targetRoot"
    }
    else {
        Write-Host "[!] LivePortrait weights source missing; mirror skipped"
    }
}
else {
    Write-Host "[*] Skipping LivePortrait download"
}

# DECA prepare
if (!$SkipDecaPrepare) {
    Write-Host "[*] Preparing DECA mirror paths"

    & $Python -c "from backend.local_models.deca_prepare import prepare_deca_repo_data; import json; print(json.dumps(prepare_deca_repo_data(), indent=2))"

    Write-Host "[!] DECA note:"
    Write-Host "    If deca_model.tar or generic_model.pkl is missing, place them here:"
    Write-Host "    local_models\deca_data\weights\deca_model.tar"
    Write-Host "    local_models\deca_data\FLAME2020\generic_model.pkl"
    Write-Host "    They may require license-compliant manual download."
}
else {
    Write-Host "[*] Skipping DECA prepare"
}

# Optional AlbedoGAN / Realistic Generative 3D Face provider
Write-Host "[*] Optional AlbedoGAN provider check"

if (!(Test-Path "local_models\Towards-Realistic-Generative-3D-Face-Models")) {
    Write-Host "[!] AlbedoGAN repo not found."
    Write-Host "    Optional future provider."
    Write-Host "    Windows clone may fail because upstream contains invalid Windows filename characters."
    Write-Host "    Do not block the pipeline on this provider."
}
else {
    Write-Host "[+] AlbedoGAN repo found."
}

Write-Host "[*] Checking model statuses"

& $Python -c "from backend.local_models.sam_aging import get_sam_aging_status; import json; print(json.dumps(get_sam_aging_status(), indent=2))"
& $Python -c "from backend.local_models.liveportrait_expression import get_liveportrait_expression_status; import json; print(json.dumps(get_liveportrait_expression_status(), indent=2))"
& $Python -c "from backend.local_models.plugin_validator import validate_all_plugins; import json; print(json.dumps(validate_all_plugins(), indent=2))"

Write-Host "[+] Model setup complete"
