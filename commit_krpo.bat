@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "feat(krpo): autonomous boost recoverability, 150 km/h unboosted & 240 km/h boost calibration, self-testing braking zones"
echo Done!
pause
