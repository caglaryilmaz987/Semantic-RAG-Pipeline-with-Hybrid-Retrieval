Set WshShell = CreateObject("WScript.Shell")
' Starts the Streamlit application in silent background mode without any console windows
WshShell.Run "cmd /c python -m streamlit run app.py --client.toolbarMode=viewer --global.developmentMode=false", 0, False
