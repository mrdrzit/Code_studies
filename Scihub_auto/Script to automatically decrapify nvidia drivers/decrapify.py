import os
import shutil

cleaned_version = ["Display.Driver", "HDAudio", "MSVCRT", "PhysX", "NvDLISR", "NvApp"]

path = os.getcwd()
dirs = os.listdir(path)

for folder in dirs:
    if folder in cleaned_version:
        continue
    else:
        path_current = os.path.abspath(os.path.join(path, folder))
        if os.path.isdir(path_current):
            shutil.rmtree(path_current, ignore_errors=False, onerror=None)
            print(f"Deleted {folder}")