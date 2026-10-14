# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "fix(krpo): eliminate false braking on flat-out sweepers/kinks, exact 10.0 arcade game physics parity (150 unboosted / 240 boosted)"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green
