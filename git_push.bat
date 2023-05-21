git add .
@echo off
set /p commits=ÇëÊäÈëcommits:
git commit -m "%commits%"
git push rc master
pause