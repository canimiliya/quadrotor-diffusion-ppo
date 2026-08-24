param([int]$ValPid = 5832)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$log = Join-Path $root 'artifacts\s8r4\post_after_val.log'
Set-Location $root
$env:PYTHONPATH = "$root;$root\src"
Add-Content $log ((Get-Date).ToString('s') + ' waiting for VAL pid=' + $ValPid)
Wait-Process -Id $ValPid -ErrorAction SilentlyContinue
Add-Content $log ((Get-Date).ToString('s') + ' VAL ended; finalizing')
& 'D:\anaconda\Scripts\conda.exe' run -n smd-blackwell python -m scripts.finalize_s8r4 *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python -m scripts.plot_s8r4 *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python -m scripts.verify_s8r4 *>> $log
& 'D:\anaconda\Scripts\conda.exe' run -n dp_quad_py310 python -m scripts.write_s8r4_report *>> $log
Add-Content $log ((Get-Date).ToString('s') + ' post VAL pipeline complete')
