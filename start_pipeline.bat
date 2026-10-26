@echo off
setlocal
pushd "%~dp0"
set "PYTHON=.venv\Scripts\pythonw.exe"
if not exist "%PYTHON%" (
    set "PYTHON=.venv\Scripts\python.exe"
)
if not exist "%PYTHON%" (
    echo ERROR: No Python interpreter found in .venv\Scripts.
    echo Create the virtual environment first with: python -m venv .venv
    pause
    popd
    endlocal
    exit /b 1
)
"%PYTHON%" -c "from continuous_zabbix_pipeline import start_continuous_pipeline; print(start_continuous_pipeline('http://192.168.100.60/zabbix/api_jsonrpc.php', 'c37bbc02ded0e038c8a16c9441a9d1a4f52c5720c49f3c8a4a7e2051f1192830', metric_keywords=['CPU','Memory','Disk','Network'], interval_seconds=300))"
popd
endlocal
