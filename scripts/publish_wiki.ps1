# Publish docs/*.md to the GitHub Wiki for 2111gt/gt_harness.
#
# GitHub only creates the wiki.git remote after the FIRST page is created
# once in the web UI (Wiki → Create the first page → Save, even empty Home).
# After that, this script can push all pages from docs/.
#
# Usage (from repo root):
#   .\scripts\publish_wiki.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Docs = Join-Path $Root "docs"
$WikiDir = Join-Path $env:TEMP "gt_harness_wiki_publish"
$WikiRemote = "https://github.com/2111gt/gt_harness.wiki.git"

if (-not (Test-Path $Docs)) { throw "docs/ not found at $Docs" }

$token = $env:GH_TOKEN
if (-not $token) {
  $token = (gh auth token 2>$null)
}
if (-not $token) { throw "Run: gh auth login   (or set GH_TOKEN)" }

if (Test-Path $WikiDir) { Remove-Item -Recurse -Force $WikiDir }

Write-Host "Cloning wiki remote..."
$clone = git clone $WikiRemote $WikiDir 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "Wiki git remote does not exist yet (normal for a never-used wiki)."
  Write-Host "One-time setup:"
  Write-Host "  1. Open https://github.com/2111gt/gt_harness/wiki"
  Write-Host "  2. Click 'Create the first page' and Save (title: Home)"
  Write-Host "  3. Re-run: .\scripts\publish_wiki.ps1"
  Write-Host ""
  Write-Host "Until then, docs are available in the repo: docs/Home.md"
  exit 1
}

Copy-Item (Join-Path $Docs "*.md") $WikiDir -Force
# GitHub wiki uses Home.md as the landing page (already named correctly)

Push-Location $WikiDir
try {
  git config user.email "pfnagy@users.noreply.github.com"
  git config user.name "pfnagy"
  git add -A
  $status = git status --porcelain
  if (-not $status) {
    Write-Host "Wiki already up to date."
    exit 0
  }
  git commit -m "Sync wiki from docs/"
  git remote set-url origin "https://x-access-token:${token}@github.com/2111gt/gt_harness.wiki.git"
  git push origin HEAD
  Write-Host "Wiki published: https://github.com/2111gt/gt_harness/wiki"
} finally {
  Pop-Location
}
