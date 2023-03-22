import glob
import subprocess
from pathlib import Path

files = glob.glob("../*.py")

for f in files:
   path = Path(f) 
   if path.stem != "make":
       subprocess.call(f"pyinstrument -r html -o ./pyinstrument-{path.stem}.html {f}", shell=True)

