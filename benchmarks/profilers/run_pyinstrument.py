import subprocess
import sys 

sys.path.append('../')
from make import AVAILABLE_BENCHMARKS

if __name__ == "__main__":
    for name, filename in AVAILABLE_BENCHMARKS.items():
       path = f"../{filename}"
       subprocess.call(f"pyinstrument -r html -o ./pyinstrument-{name}.html {path}", shell=True)
    
