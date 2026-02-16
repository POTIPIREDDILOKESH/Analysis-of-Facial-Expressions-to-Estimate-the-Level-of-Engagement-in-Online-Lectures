@echo off
set OPENFACE=..\OpenFace\FeatureExtraction.exe
set OUTPUT=..\openface_output

for /r ..\videos %%f in (*.avi) do (
    echo Processing %%f
    %OPENFACE% -f "%%f" -out_dir "%OUTPUT%"
)

echo ALL VIDEOS PROCESSED!
pause
