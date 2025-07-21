import numpy as np
import pandas as pd
from scipy.signal import welch
from os.path import join, exists
from tqdm import tqdm
from glob import glob
import tifffile

# Loading

def load_data(data_folder, metadata_path):
    meta_file = pd.read_csv(metadata_path)
    folders = [join(data_folder, folder) for folder in meta_file["Sample"].values]

    samples = {}
    for i, folder in tqdm(enumerate(folders)):

        sample_metadata = meta_file.iloc[i]
        sample_name = sample_metadata["Sample"]
        samples[sample_name] = {}
        samples[sample_name]["metadata"] = {}
        for col in meta_file.columns:
            samples[sample_name]["metadata"][col] = sample_metadata[col]

        suite2p_path = join(folder, "plane0/")
        if not exists(suite2p_path):
            suite2p_path = join(folder, "suite2p/plane0/")
        F = np.load(join(suite2p_path, "F.npy"))
        stat = np.load(join(suite2p_path, "stat.npy"), allow_pickle=True)
        is_cell = np.load(join(suite2p_path, "iscell.npy"))
        cells2keep = is_cell[:, 0].astype(bool)

        F = F[cells2keep, :]
        stat = stat[cells2keep]

        samples[sample_name]["F"] = F 
        samples[sample_name]["stat"] = stat

        # Check for mask file

        mask_path = glob(join(folder, "*mask.tiff"))

        if len(mask_path) > 0:
            mask_path = mask_path[0]
            if exists(mask_path):
                mask = tifffile.imread(mask_path)
                samples[sample_name]["mask"] = mask

                mask_thresh = mask > 0
                in_organoid = np.zeros(len(stat), dtype=bool)
                for nn in range(len(stat)):
                    coords = stat[nn]["med"]
                    if mask_thresh[coords[0], coords[1]] == 1:
                        in_organoid[nn] = 1

                samples[sample_name]["in_mask"] = in_organoid

            else:
                samples[sample_name]["mask"] = None
        else:
            samples[sample_name]["mask"] = None


        

    return samples


# Trace SNR
def snr(trace, fs=7.5, signal_range=(0, 0.2), noise_range=(0.2, 0.4)):
    f, Pxx_den = welch(trace, fs, nperseg=1024)
    signal_density = Pxx_den[(f>signal_range[0]) & (f<signal_range[1])].mean()
    noise_density = Pxx_den[(f>noise_range[0]) & (f<noise_range[1])].mean()
    return signal_density/noise_density


# Shuffling
def circular_shuffle(traces):
    traces_shuffled = traces.copy()
    T = traces.shape[1]
    for nn in range(traces.shape[0]):
        traces_shuffled[nn, :] = np.roll(traces_shuffled[nn, :], np.random.randint(-T, T))
    
    return traces_shuffled


def normalized_correlation(traces, n_shuff=100):
    actual = np.corrcoef(traces)

    corr_shuf = np.zeros((actual.shape[0], actual.shape[1], n_shuff))
    for ss in range(n_shuff):
        traces_shuff = circular_shuffle(traces)
        corr_shuf[:, :, ss] = np.corrcoef(traces_shuff)
    
    corr_norm = (actual - np.mean(corr_shuf, axis=2))/np.std(corr_shuf, axis=2)
    return (corr_norm)