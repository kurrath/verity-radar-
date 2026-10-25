@echo off
REM VERITY//RADAR launcher (Windows) - no installs needed, only Python 3.8+
cd /d "%~dp0"
where python >nul 2>nul && ( python server.py & goto :eof )
where py >nul 2>nul && ( py -3 server.py & goto :eof )
echo Python 3.8+ is required: https://www.python.org/downloads/
pause
