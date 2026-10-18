# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "feat(rl): achieve 69.75s circuit record with Dueling DDQN, 12-action kinematics, and momentum retention"
git push
Write-Host "`nKRPO repository commit and push complete!" -ForegroundColor Green

