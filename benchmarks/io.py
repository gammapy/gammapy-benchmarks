# To check the reading/writing performance of DL3 data
import logging
import numpy as np
import time
import yaml
import os
from gammapy.data import DataStore
from gammapy.maps import Map

N_OBS = int(os.environ.get("GAMMAPY_BENCH_N_OBS", 10))

def run_benchmark():
    info = {"n_obs": N_OBS}

    t = time.time()

    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    OBS_ID = 110380
    obs_ids = OBS_ID * np.ones(N_OBS)
    observations = data_store.get_observations(obs_ids)

    info["data_loading"] = time.time() - t
    t = time.time()

    m = Map.create()
    for obs in observations:
        m.fill_events(obs.events)

    info["filling"] = time.time() - t
    t = time.time()

    m.write("survey_map.fits.gz", overwrite=True)

    info["writing"] = time.time() - t

    with open("bench.yaml", "w") as fh:
        yaml.dump(info, fh, sort_keys=False, indent=4)


if __name__ == "__main__":
    format = "%(filename)s:%(lineno)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=format)
    logging.info(f"Running io.py with N_OBS = {N_OBS}")
    logging.info(f"cwd = {os.getcwd()}")
    run_benchmark()
