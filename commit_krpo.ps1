# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "fix(krpo): butter-smooth ghost car replay with Hermite spline interpolation, hardware clock sync, and lane separation"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green

