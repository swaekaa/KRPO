@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "feat(krpo): 40-estimation simultaneous swarm replay, progressive evolutionary field, multi-car camera tracking"
echo Done!
pause

