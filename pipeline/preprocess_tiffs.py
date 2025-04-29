import numpy as np
import tifffile
import os
import argparse

from scipy.ndimage.filters import gaussian_filter1d

FS = 7.55
BASELINE_LIMIT = int(3.5*60*FS)
STIM_LIMIT = int(3.5*60*FS)

def stim_timepoints(field_intensity):
    binary_arr = field_intensity < 3
    timepoints = {"stim_start":[], "stim_end":[]}
    for ix, tt in enumerate(range(len(binary_arr)-1)):
        if (binary_arr[tt] == 0) and (binary_arr[tt+1]==1):
            timepoints["stim_start"].append(ix)
        if (binary_arr[tt] == 1) and (binary_arr[tt+1]==0):
            timepoints["stim_end"].append(ix+5)
    return timepoints


def preprocess_tiffs(data_folder, output_folder, chop_video=True): 
    """
    Preprocesses TIFF files in the specified folder by applying a Gaussian filter and saving the results.       
    Args:                                                                                   
        folder (str): Path to the folder containing TIFF files.
    """
    files = os.listdir(data_folder)
    print(files)
    files = [f for f in files if f.endswith(".tiff")]  #Expecting .tiff files not .tif
    for i, file in enumerate(files):
        print(file)
        path = os.path.join(data_folder, file)
        save_name = str.replace(file, ".tiff", "_proc.tiff")
        video_folder = os.path.join(output_folder, str.replace(file, ".tiff", ""))

        if not os.path.exists(video_folder):
            os.mkdir(video_folder)
        else:
            continue

        vid = tifffile.imread(path)

        if chop_video:
            field_intensity = np.mean(np.mean(vid, axis=2), axis=1)
            tps = stim_timepoints(field_intensity)

            baseline_segment = vid[:tps["stim_start"][0], :, :]
            baseline_segment = baseline_segment[-BASELINE_LIMIT:, :, :]

            if len(tps["stim_start"]) == 2:
                stim_segment = vid[tps["stim_end"][0]:tps["stim_start"][1], :, :]
            else:
                stim_segment = vid[tps["stim_end"][0]:, :, :]

            stim_segment = stim_segment[:STIM_LIMIT, :, :]
            vid_concat = np.concatenate((baseline_segment, stim_segment), axis=0)
        else:
            vid_concat = vid

        vid_filt = gaussian_filter1d(vid_concat, FS/2, axis=0)

        tifffile.imsave(os.path.join(video_folder, save_name), vid_filt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--chop", "-c", action="store_true")
    args = parser.parse_args()

    print(args.chop)

    output_folder = os.path.join(args.folder, "processed")

    if not os.path.exists(output_folder):
        print("Creating output folder")
        os.mkdir(output_folder)
    else:
        print(f"Found existsing output folder at {output_folder}")
    
    preprocess_tiffs(args.folder, output_folder, chop_video=args.chop)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()