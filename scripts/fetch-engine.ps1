$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
& $pythonPath (Join-Path $projectRoot 'engine/fetch.py') $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Source fetch failed' }
Write-Output 'Pinned sources ready. Build the native engine in Docker or Linux; see README.'
