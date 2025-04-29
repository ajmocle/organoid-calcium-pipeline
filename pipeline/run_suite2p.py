import os
import pathlib
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

import suite2p

import argparse

def run_suite2p(data_folder, ops):

    ops['save_folder'] = data_folder
    ops['save_path0'] = data_folder
    tiff_file = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".tiff")]
    db = {'data_path': data_folder, 'temp_folder': TemporaryDirectory().name, 'tiff_list': tiff_file}

    _ = suite2p.run_s2p(ops=ops, db=db)

    suite2p_path = os.path.join(data_folder, "suite2p")
    if os.path.exists(suite2p_path):
        print(f"Cleaning up binary copy folder in {suite2p_path}")
        plane0_path = os.path.join(suite2p_path, "plane0")
        os.remove(os.path.join(plane0_path, "data.bin"))
        pathlib.Path.rmdir(pathlib.Path(plane0_path))
        pathlib.Path.rmdir(pathlib.Path(suite2p_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--ops", "-o", default="")
    args = parser.parse_args()

    ops_path = args.ops
    ops = np.load(ops_path, allow_pickle=True).item()

    processed_folder = args.folder
    data_folder_list = [os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if os.path.isdir(os.path.join(processed_folder, f))]
    print(data_folder_list)
    for data_folder in data_folder_list:
        print(f"Processing {data_folder}")
        run_suite2p(data_folder, ops)
        print(f"Finished processing {data_folder}")

if __name__ == "__main__":
    main()