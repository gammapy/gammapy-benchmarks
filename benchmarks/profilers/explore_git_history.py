import logging
import git
import io
import codecs
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import datetime
import subprocess
import sys
sys.path.append('../')
from make import AVAILABLE_BENCHMARKS

log = logging.getLogger()

def get_benchmark_history(bench, path):
    """Build list of benchmark times accross git repo."""
    log.info(f"Getting {bench} history in {path} repo.")
    repo = git.Repo(path)

    file_path = "benchmarks/results/" + bench + "/bench.yaml"
    log.info(f"Exploring commit history of {file_path}.")

    commits = list(repo.iter_commits('--all', paths=file_path))
    log.info(f"Found {len(commits)} commits.")

    dates = []
    bench_times = []
    labels = []
    
    for commit in commits:
        author = commit.committer.name
        commit_time = commit.committed_datetime
        sha = commit.hexsha
        log.debug(f"Committed by {author} on {commit_time} with sha {sha}")

        if author == "GitHub Actions":
            blob = commit.tree / file_path

            StreamReader = codecs.getreader('utf-8')
            f = StreamReader(io.BytesIO(blob.data_stream.read()))

            times = []
            labels = []
            for line in f.readlines():
                label, time = line.split(sep=':')
                times.append(float(str(time)))
                labels.append(str(label))

            dates.append(commit_time)
            bench_times.append(times)

    times_array = np.vstack(bench_times).T
    return dates, times_array, labels, commits

def plot_benchmark_history(bench, dates, times_array, labels=None): 
    """Plot the time history and get peak dates mask."""
    log.info(f"Plotting benchmark history of {bench}.")
    size = times_array.shape[0]-1
    dates=np.array(dates)[::-1]

    mask_peaks=np.zeros(dates.shape, dtype=bool)
    fig, axs = plt.subplots(size,figsize=(12,10), sharex=True)
    for idx, ax in enumerate(axs):
        label=None
        if labels is not None:
           label = labels[idx+1] 
        xtmp = times_array[idx+1][::-1]
        width = 7 #days
        ax.step(dates, xtmp, 'k-', alpha=0.3, where='pre', label=label)
        ax.set_ylabel("time, s")
        
        with np.errstate(divide='ignore', invalid='ignore'):
            xdiff = forward_backward_difference(dates, xtmp, width)

        thr = 1 #std
        percent = 0.3
        xmax = ndi.maximum_filter(xdiff, size=size, mode="constant")
        safe_cut = (xdiff > thr * np.nanstd(xdiff)) & (xdiff>percent*xtmp)
        mask = (xdiff == xmax) & safe_cut
        peaks = dates[mask]
        mask_peaks |= mask
        yl = ax.get_ylim()
        ax.plot([dates[0],dates[-1]], [0,0], 'k--')
        ax.plot(dates, xdiff, 'r-')
        for peak in peaks:
            ax.plot([peak, peak], yl, 'b--')

        ax.legend()
    ax.set_xlabel("date")
    axs[0].set_title(bench)
    plt.savefig(bench+"_history.png") 
    return mask_peaks[::-1]

def forward_backward_difference(dates, values, width):
    """Differrence between the mean in `width` days forward and backward"""
    delta = dates - dates[:,None] 

    before=datetime.timedelta(days=-width)
    present = datetime.timedelta(days=0)
    after=datetime.timedelta(days=width)

    backward = values * (delta>before) * (delta<present)
    forward =  values * (delta>present) * (delta<after)
    backward = np.sum(backward, axis=1)/np.sum(backward>0, axis=1)
    forward = np.sum(forward, axis=1)/np.sum(forward>0, axis=1)
    return  forward - backward
    

def get_merge_log(path, name):
    subprocess.call(f"git -C {path} log --merges --first-parent master --max-count=500 --pretty=format:'%H, %ad, %s' --date=iso-strict > {name}_merges.log", shell=True)
    data=np.genfromtxt(f"{name}_merges.log", delimiter=",", dtype=None, encoding=None, comments=None)
    commits = data[:,0]
    dates = np.array([datetime.datetime.fromisoformat(date.strip()) for date in data[:,1]])
    messages = data[:,-1]
    for k, message in enumerate(messages):
        messages[k] = message[20:]
    return commits, dates, messages


def run_profiler(gammapy, benchmarks, bench, dates, commits=None):
    for k, date in enumerate(dates):
        if commits is None:
            subprocess.call(f"git -C {gammapy} checkout " + "'master@{" + date +"}'", shell=True)
            subprocess.call(f"git -C {benchmarks} checkout " + "'master@{" + date +"}'", shell=True)
            output =  subprocess.check_output(f"git  -C {gammapy} rev-parse HEAD", shell=True)
            gammapy_commit = output.decode('utf-8')[:-1]
        else:
            gammapy_commit = commits[k]
            subprocess.call(f"git -C {gammapy} checkout {gammapy_commit}", shell=True)
            subprocess.call(f"git -C {benchmarks} checkout " + "'master@{" + date +"}'", shell=True)
            
        output = f"{benchmarks}/benchmarks/profilers/pyinstrument_{bench}_{gammapy_commit}_{date[:10]}.html"
        subprocess.call(f"pyinstrument -r html -o output {benchmarks}/benchmarks/{bench}.py", shell=True)

#%%


if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO)

   benchmarks = "../../"

   for bench in AVAILABLE_BENCHMARKS.keys():
       log.info(bench)
       try:
           dates, bench_times, labels, commits = get_benchmark_history(bench, benchmarks)
           mask = plot_benchmark_history(bench, dates, bench_times, labels)
           ind_peak = np.where(mask)[0]
           for ind in ind_peak:
               log.info("Peak : ", dates[ind])
       except:
           log.info("skipped")
           continue

