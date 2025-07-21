import numpy as np
from skimage.morphology import closing, disk
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("plane0_path", type=str, help="Path to the data folder containing the plane0 directory.")
args = parser.parse_args()  

def pixels_to_img(ypix, xpix, lam, shape):
    """
    Convert pixel coordinates to an image array.
    """
    img = np.zeros(shape, dtype=bool)
    for i, (y, x) in enumerate(zip(ypix, xpix)):
        img[y, x] = lam[i]
    return img

def img_to_pixels(img):
    """
    Convert an image array to pixel coordinates.
    """
    ypix, xpix = np.nonzero(img)
    lam = img[ypix, xpix]
    return ypix, xpix, lam


def postprocess_sfps(plane0_path):
    stat = np.load(os.path.join(plane0_path, "stat.npy"), allow_pickle=True)
    stat_new = stat.copy()

    for i in tqdm(range(len(stat_new))):
        ypix, xpix, lam = stat_new[i]["ypix"], stat_new[i]["xpix"], stat_new[i]["lam"]
        img = pixels_to_img(ypix, xpix, lam, (512, 512))
        closing_img = closing(img, disk(3))
        ypix_new, xpix_new, lam_new = img_to_pixels(closing_img)
        stat_new[i]["ypix"] = ypix_new
        stat_new[i]["xpix"] = xpix_new
        stat_new[i]["lam"] = lam_new

    os.mkdir(os.path.join(plane0_path, "original_files"))
    os.rename(os.path.join(plane0_path, "stat.npy"), os.path.join(plane0_path, "original_files", "stat.npy"))
    np.save(os.path.join(plane0_path, "stat.npy"), stat_new)

if __name__ == "__main__":
    postprocess_sfps(args.plane0_path)
    print("Postprocessing complete.")