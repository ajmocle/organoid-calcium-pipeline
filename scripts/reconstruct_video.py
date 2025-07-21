import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage.filters import gaussian_filter1d

def reconstruct_video(data_folder, tiff_path):
    F_path = os.path.join(data_folder, "F.npy")
    F = np.load(F_path)
    stat_path = os.path.join(data_folder, "stat.npy")   
    stat = np.load(stat_path, allow_pickle=True)
    is_cell_path = os.path.join(data_folder, "iscell.npy")
    is_cell = np.load(is_cell_path)
    cells2keep = is_cell[:, 0].astype(bool)
    F = F[cells2keep, :]
    F = gaussian_filter1d(F, sigma=7.55, axis=1)  # Smooth the traces
    stat = stat[cells2keep]

    # Load the tiff file    
    vid = tifffile.imread(tiff_path)

    n_cells = F.shape[0]

    recon = np.zeros(vid.shape, dtype=np.uint8)
    for n in range(n_cells):
        y, x = stat[n]["ypix"], stat[n]["xpix"]
        for t in range(vid.shape[0]):
            recon[t, y, x] = F[n, t]

    vid_recon = np.concatenate((vid, recon), axis=2)
    savename = str.replace(tiff_path, ".tiff", "_recon.tiff")
    tifffile.imwrite(savename, vid_recon)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--tiff-path", type=str)
    args = parser.parse_args()

    reconstruct_video(args.data_path, args.tiff_path)
    print("Reconstruction complete.")
    
if __name__ == "__main__":
    main()  