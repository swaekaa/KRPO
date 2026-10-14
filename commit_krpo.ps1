# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "feat(krpo): 40-estimation simultaneous swarm replay, progressive evolutionary field, multi-car camera tracking"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green

