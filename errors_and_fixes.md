# Known Errors and Fixes
This file lists project related errors and possible fixes.

## Fixed Errors
### Python
- Activation of virtualenv `.\venv\Scripts\activate` returns the error 
```
.\venv\Scripts\activate : Die Datei "C:\Users\KrebsJ\Projekte\llm-pipeline-konzmgt\azure-concept-summary\venv\Scripts\Activate.ps1" kann nicht geladen werden, da die Ausführung von Skripts auf diesem System deaktiviert ist.  
Weitere Informationen finden Sie unter "about_Execution_Policies" (https:/go.microsoft.com/fwlink/?LinkID=135170).
```
**Fix:** Execute `Set-ExecutionPolicy Unrestricted -Scope Process` and try again



## Open Errors