@echo off
setlocal
pushd "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    set "PYTHON=.venv\Scripts\pythonw.exe"
)
if not exist "%PYTHON%" (
    echo ERROR: No Python interpreter found in .venv\Scripts.
    echo Create the virtual environment first with: python -m venv .venv
    pause
    popd
    endlocal
    exit /b 1
)
"%PYTHON%" -c "from continuous_zabbix_pipeline import stop_continuous_pipeline; print(stop_continuous_pipeline())"
popd
endlocal
