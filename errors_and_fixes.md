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
- Installing any package via pip returns the error
```
Could not fetch URL https://pypi.org/simple/openai/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/openai/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1000)'))) - skipping
ERROR: Could not find a version that satisfies the requirement openai (from versions: none)
ERROR: No matching distribution found for openai
```
**Fix:** Add `--trusted-host` param to the install command, e.g. `pip install --trusted-host pypi.org openai`
- Running a python script that uses a https request returns the error
```
httpcore.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain 
```
This is caused by the corporate proxy/VPN using a self-signed certificate.
**Fix:** 
1. Install certifi `pip install certifi`
2. Open the Azure API-endpoint in your Browser and download the certificate chain, e.g. `cognitiveservices-azure-com-zertifikatskette.pem`
3. Copy the text content of the certificate chain .pem-file and append it to the cert.pem-file python uses. You can find the location of the file with `print(certifi.where())`
## Open Errors