@echo off
setlocal
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=ga_log.txt
echo [GA] Running optimizer. Logs: %CD%\%LOG%
echo [GA] This can take several minutes depending on population/generations.

if exist ".venv\\Scripts\\python.exe" (
  .venv\\Scripts\\python.exe -u -m lottogogo.tuning.ga_optimizer --csv history.csv --train-end 1000 --val-end 1050 --population 20 --generations 10 --output data/optimized_weights_no_hmm.json > %LOG% 2>&1
) else (
  uv run python -u -m lottogogo.tuning.ga_optimizer --csv history.csv --train-end 1000 --val-end 1050 --population 20 --generations 10 --output data/optimized_weights_no_hmm.json > %LOG% 2>&1
)

echo DONE (exit=%ERRORLEVEL%)>> %LOG%
echo [GA] Finished. Exit=%ERRORLEVEL%. See %LOG%
exit /b %ERRORLEVEL%
