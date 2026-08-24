param([int]$TrainingPid = 38444)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$log = Join-Path $root 'artifacts\s8r4\post_pipeline.log'
Add-Content $log ((Get-Date).ToString('s') + ' waiting for training pid=' + $TrainingPid)
Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
Add-Content $log ((Get-Date).ToString('s') + ' training ended; starting VAL')
& 'D:\anaconda\Scripts\conda.exe' run -n smd-blackwell python scripts/parallel_s8r4_val.py --workers 6 *>> $log
Add-Content $log ((Get-Date).ToString('s') + ' VAL ended; finalizing')
& 'D:\anaconda\Scripts\conda.exe' run -n smd-blackwell python scripts/finalize_s8r4.py *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python scripts/plot_s8r4.py *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python scripts/verify_s8r4.py *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python scripts/write_s8r4_report.py *>> $log
Add-Content $log ((Get-Date).ToString('s') + ' post pipeline complete')
