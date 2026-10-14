# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "feat(krpo): autonomous boost recoverability, 150 km/h unboosted & 240 km/h boost calibration, self-testing braking zones"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green
