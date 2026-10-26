@echo off
setlocal enabledelayedexpansion

rem Determine the installation directory from this batch file's location.
set "INSTALL_DIR=%~dp0"
set "SHORTCUT=%USERPROFILE%\Desktop\Zabbix ML.lnk"

echo Removing desktop shortcut if present...
if exist "%SHORTCUT%" del /f /q "%SHORTCUT%"

echo Scheduling removal of installation directory...
cd /d "%TEMP%"
(
    echo @echo off
    echo timeout /t 2 /nobreak ^>nul
    echo rd /s /q "%INSTALL_DIR%"
    echo del /f /q "%%~f0"
) > "%TEMP%\zabbix_ml_uninstall_cleanup.bat"

start "" /b cmd /c "%TEMP%\zabbix_ml_uninstall_cleanup.bat"
echo Uninstall started. The installation directory will be removed shortly.
pause
