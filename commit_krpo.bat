@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "feat(krpo): continuous open-ended evolutionary policy, sub-1:16 breakthrough, dynamic apex momentum retention"
echo Done!
pause
