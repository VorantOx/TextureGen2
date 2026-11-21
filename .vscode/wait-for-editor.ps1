$ErrorActionPreference = "SilentlyContinue"
$deadline = (Get-Date).AddSeconds(45)
$ok = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8765/api/jobs" -TimeoutSec 1
        if ($r.StatusCode -eq 200) {
            $ok = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 250
    }
}
if (-not $ok) {
    Write-Error "OTWN editor did not answer on http://127.0.0.1:8765/ within 45 seconds."
    exit 1
}
Write-Host "OTWN editor is ready on http://127.0.0.1:8765/"
exit 0
