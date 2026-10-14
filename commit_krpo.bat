@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "fix(krpo): eliminate false braking on flat-out sweepers/kinks, exact 10.0 arcade game physics parity (150 unboosted / 240 boosted)"
echo Done!
pause
