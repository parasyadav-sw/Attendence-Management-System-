@echo off
echo Starting ParaScan - Face Recognition Attendance System...

REM Start Flask server in background
start /B python app.py

REM Wait for server to start
timeout /t 4 /nobreak >nul

REM Open browser
start http://localhost:5000

echo.
echo Server is running! Browser opening...
echo Close this window to stop the server.
pause
