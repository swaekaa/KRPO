# Stage and commit all changes in KRPO repository
Set-Location -Path $PSScriptRoot
git add -A
git commit -m "fix(krpo): declare f1 and f2 frame aliases in sampleReplayState Hermite interpolation"
Write-Host "`nKRPO repository commit complete!" -ForegroundColor Green

