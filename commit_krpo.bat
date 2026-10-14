@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "fix(krpo): butter-smooth ghost car replay with Hermite spline interpolation, hardware clock sync, and lane separation"
echo Done!
pause

