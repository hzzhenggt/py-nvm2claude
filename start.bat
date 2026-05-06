@echo off
chcp 65001
setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
echo %SCRIPT_DIR%

set TAB1=cmd /k "cd /d %SCRIPT_DIR% && echo %cd% && python pynvmproxy.py"

set TAB2=cmd /k "cd /d %SCRIPT_DIR% && type proxy.log "

wt.exe new-tab -p "Command Prompt" %TAB1%; split-pane -p "Command Prompt" %TAB2%

endlocal
