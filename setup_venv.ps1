param(
    [string]$VenvDir = ".venv",
    [string]$RequirementsFile = "requirements.txt"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-Python312 {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$PrefixArgs = @()
    )

    try {
        $version = & $Exe @($PrefixArgs + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"))
        return ($version.Trim() -eq "3.12")
    }
    catch {
        return $false
    }
}

function Resolve-Python312 {
    $candidates = @(
        @{ Exe = "py"; PrefixArgs = @("-3.12") },
        @{ Exe = "python3.12"; PrefixArgs = @() },
        @{ Exe = "python"; PrefixArgs = @() }
    )

    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) {
            continue
        }

        if (Test-Python312 -Exe $candidate.Exe -PrefixArgs $candidate.PrefixArgs) {
            return $candidate
        }
    }

    throw "Khong tim thay Python 3.12. Vui long cai Python 3.12 truoc khi chay script."
}

Write-Host "[1/4] Dang tim Python 3.12..."
$pythonCmd = Resolve-Python312

Write-Host "[2/4] Tao virtual environment: $VenvDir"
& $pythonCmd.Exe @($pythonCmd.PrefixArgs + @("-m", "venv", $VenvDir))

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Khong tim thay python trong venv: $venvPython"
}

if (-not (Test-Path $RequirementsFile)) {
    throw "Khong tim thay file requirements: $RequirementsFile"
}

Write-Host "[3/4] Nang cap pip/setuptools/wheel..."
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[4/4] Cai dependencies tu $RequirementsFile ..."
& $venvPython -m pip install -r $RequirementsFile

Write-Host ""
Write-Host "Hoan tat. Cach kich hoat venv:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Chay server:"
Write-Host "  uvicorn server:app --reload"