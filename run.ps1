# Workaround when Application Control blocks .venv\Scripts\python.exe.
# Uses the base interpreter from .venv\pyvenv.cfg with this venv's packages.
# Usage: .\run.ps1 -m src.bots.multimodal_trader --mode swing

$root = $PSScriptRoot
$cfgPath = Join-Path $root ".venv\pyvenv.cfg"
if (-not (Test-Path $cfgPath)) {
    Write-Error "Missing .venv\pyvenv.cfg. Create the venv first."
    exit 1
}

$homeLine = Get-Content $cfgPath | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
if (-not $homeLine) {
    Write-Error "Could not read 'home' from .venv\pyvenv.cfg"
    exit 1
}
$pyHome = ($homeLine -split '=', 2)[1].Trim()
$py = Join-Path $pyHome "python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Python not found: $py"
    exit 1
}

$venv = Join-Path $root ".venv"
$sitePackages = Join-Path $venv "Lib\site-packages"
$scripts = Join-Path $venv "Scripts"

$env:VIRTUAL_ENV = $venv
$env:PYTHONPATH = $sitePackages
if ($env:PYTHONHOME) { Remove-Item Env:PYTHONHOME }
$env:PATH = "$scripts;$env:PATH"
Set-Location $root

& $py @args
