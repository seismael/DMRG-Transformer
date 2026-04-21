#Requires -Version 5.1
<#
.SYNOPSIS
    Full quality gate: lint, type-check, and test.

.DESCRIPTION
    Runs ruff (lint + format), mypy (static types), and pytest (all gates
    including the @slow benchmark smoke test). Fails fast on first error.

.EXAMPLE
    pwsh -File scripts/check.ps1
#>

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $env:PYTHONPATH = 'src'

    Write-Host '==> ruff check' -ForegroundColor Cyan
    python -m ruff check src tests
    if ($LASTEXITCODE) { throw "ruff check failed ($LASTEXITCODE)" }

    Write-Host '==> ruff format --check' -ForegroundColor Cyan
    python -m ruff format --check src tests
    if ($LASTEXITCODE) { throw "ruff format check failed ($LASTEXITCODE)" }

    Write-Host '==> mypy' -ForegroundColor Cyan
    python -m mypy src
    if ($LASTEXITCODE) { throw "mypy failed ($LASTEXITCODE)" }

    Write-Host '==> pytest (all gates + smoke benchmark)' -ForegroundColor Cyan
    python -m pytest tests -v
    if ($LASTEXITCODE) { throw "pytest failed ($LASTEXITCODE)" }

    Write-Host 'ALL CHECKS PASSED' -ForegroundColor Green
}
finally {
    Pop-Location
}
