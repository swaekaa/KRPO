@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "feat(krpo): unconventional lines (V-line squaring, chicane cuts, trail-brake dive), stochastic RL spline exploration"
echo Done!
pause
