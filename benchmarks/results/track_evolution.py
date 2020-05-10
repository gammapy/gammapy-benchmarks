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
with open(monthly_yaml_file,"w") as stream:
    yaml.safe_dump(monthly_yaml_content, stream)

# TODO: make plot of monthly evolution and overwrite png file
