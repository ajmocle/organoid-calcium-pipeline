import os
import numpy as np
import argparse
import tifffile
from tqdm import tqdm

from readlif.reader import LifFile

def extract_tiffs_from_lif(lif_file_path, output_dir):
    """
    Extracts all TIFF images from a .lif file and saves them to the specified output directory.

    Args:
        lif_file_path (str): Path to the .lif file.
        output_dir (str): Directory where the extracted TIFF files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the .lif file
    lif = LifFile(lif_file_path)
    # Iterate through each image in the .lif file
    for image in tqdm(lif.get_iter_image(), desc="Extracting TIFFs"):
        # Construct the output file path
        file_name = image.name
        output_file_path = os.path.join(output_dir, f"{file_name}.tiff") # TODO fix naming
        # Save the image as a TIFF file
        vid = np.zeros((image.dims.t, image.dims.y, image.dims.x), dtype=np.uint8)
        for t in range(image.dims.t):
            vid[t, :, :] = image.get_frame(z=0, t=t, c=0)

        tifffile.imwrite(output_file_path, vid)
        print(f"Saved: {output_file_path}")

def main(): 
    parser = argparse.ArgumentParser(description="Extract TIFF images from a .lif file.")
    parser.add_argument("--lif_file", type=str, help="Path to the .lif file.")
    parser.add_argument("--output_dir", "-o", type=str, help="Directory to save the extracted TIFF files.")
    args = parser.parse_args()

    extract_tiffs_from_lif(args.lif_file, args.output_dir)

if __name__ == "__main__":  
    main()