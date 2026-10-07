$log = "dl_direct.log"
& ".venv\Scripts\python.exe" scripts\download_direct.py inpaint_b *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
& ".venv\Scripts\python.exe" scripts\download_direct.py inpaint_ac *>&1 | Out-File -FilePath $log -Encoding utf8 -Append
"DONE_ALL" | Out-File -FilePath $log -Encoding utf8 -Append
