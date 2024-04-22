cls
del *.exe
del *.obj
cl.exe /EHsc /I . DevProp.c /link OpenCL.lib /SUBSYSTEM:CONSOLE
DevProp.exe
