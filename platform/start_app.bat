@echo off

call D:\Leko\Anaconda\Scripts\activate.bat
cd /d D:\Leko\bladder_ai_app
call conda activate multi_task
python app.py

pause
