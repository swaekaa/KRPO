# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "feat(krpo): unconventional lines (V-line squaring, chicane cuts, trail-brake dive), stochastic RL spline exploration"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green
