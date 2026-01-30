import subprocess
import sys
sys.path.append('../')
from make import AVAILABLE_BENCHMARKS

for name, filename in AVAILABLE_BENCHMARKS.items():
   path = f"../{filename}"
   subprocess.call(f"memray run -o ./memray-{name}.bin {path} ", shell=True)
   subprocess.call(f"memray table -o ./memray-table-{name}.html ./memray-{name}.bin", shell=True)
   subprocess.call(f"memray flamegraph -o ./memray-flamegraph-{name}.html ./memray-{name}.bin", shell=True)

