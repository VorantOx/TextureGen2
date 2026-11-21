$ErrorActionPreference = "SilentlyContinue"
$conn = Get-NetTCPConnection -LocalPort 8765 -State Listen |
    Select-Object -First 1
if ($conn) {
    Stop-Process -Id $conn.OwningProcess -Force
    Start-Sleep -Milliseconds 200
    Write-Host "Stopped OTWN editor PID $($conn.OwningProcess)."
} else {
    Write-Host "Port 8765 is already free."
}
exit 0
