# Publish docs/*.md to the GitHub Wiki for 2111gt/gt_harness.
#
# Wiki remote is initialized (Home page exists). Re-run anytime after editing docs/.
#
# Usage (from repo root):
#   .\scripts\publish_wiki.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Docs = Join-Path $Root "docs"
$WikiDir = Join-Path $env:TEMP ("gt_harness_wiki_publish_" + [guid]::NewGuid().ToString("N"))
$WikiRemote = "https://github.com/2111gt/gt_harness.wiki.git"

if (-not (Test-Path $Docs)) { throw "docs/ not found at $Docs" }

$token = $env:GH_TOKEN
if (-not $token) {
  $token = (gh auth token 2>$null)
}
if (-not $token) { throw "Run: gh auth login   (or set GH_TOKEN)" }

Write-Host "Cloning wiki remote into $WikiDir ..."
# git writes progress to stderr; do not treat that as a terminating error
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
git clone $WikiRemote $WikiDir 2>&1 | ForEach-Object { Write-Host $_ }
$cloneCode = $LASTEXITCODE
$ErrorActionPreference = $prevEap

if ($cloneCode -ne 0) {
  Write-Host ""
  Write-Host "Wiki git remote not available."
  Write-Host "If this is a brand-new wiki: open https://github.com/2111gt/gt_harness/wiki"
  Write-Host "create the first page (Home), then re-run this script."
  exit 1
}

try {
  Copy-Item (Join-Path $Docs "*.md") $WikiDir -Force

  Push-Location $WikiDir
  try {
    git config user.email "2111gt@users.noreply.github.com"
    git config user.name "2111gt"
    git add -A
    $status = git status --porcelain
    if (-not $status) {
      Write-Host "Wiki already up to date."
      exit 0
    }
    git commit -m "Sync wiki from docs/"
    git remote set-url origin "https://x-access-token:${token}@github.com/2111gt/gt_harness.wiki.git"
    $ErrorActionPreference = "Continue"
    git push origin HEAD 2>&1 | ForEach-Object { Write-Host $_ }
    $pushCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($pushCode -ne 0) { throw "git push failed with exit $pushCode" }
    Write-Host "Wiki published: https://github.com/2111gt/gt_harness/wiki"
  } finally {
    Pop-Location
  }
} finally {
  # Best-effort cleanup of temp clone
  try {
    if (Test-Path $WikiDir) {
      Remove-Item -Recurse -Force $WikiDir -ErrorAction SilentlyContinue
    }
  } catch {}
}
