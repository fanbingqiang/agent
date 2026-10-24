@echo off
cd /d D:\code4\agent\backend
echo Starting flight agent backend on http://localhost:8001 ...
python -m uvicorn main:app --host 0.0.0.0 --port 8001
pause
