@echo off
chcp 65001 >nul
cd /d %~dp0
echo [窑智] 启动服务 http://127.0.0.1:8000
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
