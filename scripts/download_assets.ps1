param(
    [switch]$Force,
    [switch]$SkipAccessories,
    [switch]$SkipMakeup
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$AssetsRoot = Join-Path $Root "assets"

$Folders = @(
    "accessories/glasses",
    "accessories/earrings",
    "accessories/necklaces",
    "accessories/hair_clips",
    "accessories/hats",
    "makeup/lipstick",
    "makeup/blush",
    "makeup/eyeliner",
    "makeup/eyeshadow",
    "palettes"
)

foreach ($Folder in $Folders) {
    $Path = Join-Path $AssetsRoot $Folder
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    $Keep = Join-Path $Path ".gitkeep"
    if (!(Test-Path $Keep)) {
        New-Item -ItemType File -Force -Path $Keep | Out-Null
    }
}

# Add curated direct-download entries here only after confirming license/source.
# Example shape:
# $CuratedAssets = @(
#   @{
#     Category = "glasses"
#     Url = "https://example.org/direct-file.png"
#     FileName = "example.png"
#     License = "CC0"
#     Source = "https://example.org/license"
#   }
# )
$CuratedAssets = @()

if ($CuratedAssets.Count -eq 0) {
    Write-Host "No curated asset URLs configured yet."
    Write-Host "Folders were created. Existing files were not changed."
    exit 0
}

foreach ($Asset in $CuratedAssets) {
    $Category = [string]$Asset.Category
    $Url = [string]$Asset.Url
    $FileName = [string]$Asset.FileName

    if ($SkipAccessories -and $Category -in @("glasses", "earrings", "necklaces", "hair_clips", "hats")) {
        continue
    }

    if ($SkipMakeup -and $Category -in @("lipstick", "blush", "eyeliner", "eyeshadow")) {
        continue
    }

    if ([IO.Path]::GetExtension($FileName).ToLowerInvariant() -ne ".png") {
        Write-Warning "Skipping $FileName because only PNG assets are accepted."
        continue
    }

    $TargetDir = Join-Path $AssetsRoot "accessories/$Category"
    $Target = Join-Path $TargetDir $FileName

    if ((Test-Path $Target) -and !$Force) {
        Write-Host "Skipping existing asset: $Target"
        continue
    }

    Write-Host "Downloading $Url"
    Write-Host "License: $($Asset.License)"
    Write-Host "Source: $($Asset.Source)"
    Invoke-WebRequest -Uri $Url -OutFile $Target
}

Write-Host "Asset download complete. Review licenses before adding files to git."
