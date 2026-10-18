@echo off
echo Staging and committing all changes inside KRPO...
cd /d "%~dp0"
git add -A
git commit -m "feat(rl): achieve 69.75s circuit record with Dueling DDQN, 12-action kinematics, and momentum retention"
git push
echo Done!
pause

