#!/usr/bin/env python
"""Script run in scheduled Github actions to make daily evolution plots for benchmarks"""

from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import yaml

# get today
today = date.today()
base_filename = f"{today.year}_{str(today.month).zfill(2)}"

# read results of today
with open("results.yaml", "r") as stream:
    results = yaml.safe_load(stream)
today_results = {today.day: results}

# read monthly results
monthly_yaml_content = {}
monthly_yaml_file = f"{base_filename}.yaml"
try:
    with open(monthly_yaml_file, "r") as stream:
        monthly_yaml_content = yaml.safe_load(stream)
except FileNotFoundError:
    pass

# update monthly results
monthly_yaml_content.update(today_results)
with open(monthly_yaml_file, "w") as stream:
    yaml.safe_dump(monthly_yaml_content, stream)

# reorder
total_time = {}
memory_peak = {}
for k, w in results.items():
    memory_peak[k] = np.full(31, np.nan)
    total_time[k] = np.full(31, np.nan)
    for i in np.arange(31):
        try:
            total_time[k][i] = monthly_yaml_content[i + 1][k]["total_time"]
            memory_peak[k][i] = monthly_yaml_content[i + 1][k]["memory_peak"]
        except KeyError:
            pass

# save plots of monthly evolution
plt.title("Total time (s)")
plt.xlim(1, 31)
for k, w in results.items():
    plt.plot(np.arange(1, 32), total_time[k], label=k, linewidth=1)
plt.legend()
monthly_png_time_file = f"{base_filename}_time.png"
plt.savefig(monthly_png_time_file)

plt.clf()
plt.title("Memory peak (MB)")
plt.xlim(1, 31)
for k, w in results.items():
    plt.plot(np.arange(1, 32), memory_peak[k], label=k, linewidth=1)
plt.legend()
monthly_png_memory_file = f"{base_filename}_memory.png"
plt.savefig(monthly_png_memory_file)
