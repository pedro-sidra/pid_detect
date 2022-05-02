# %%
from black import main
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import skimage
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate templates from DigitizePID_Dataset"
    )
    parser.add_argument(
        "path",
        type=str,
        default=None,
        help="Path to DigitizePID_Dataset (default: None)",
    )

    parser.add_argument(
        "-o", "--outpath", type=str, default="./outputs/", help="Path to write outpus"
    )
    parser.add_argument(
        "-t", "--threshold", action="store_true", help="Save thresholds of image"
    )
    args = parser.parse_args()

    dbpath = Path(args.path)
    outPath = Path(args.outpath)

    if not outPath.is_dir():
        outPath.mkdir()

    imagepath = dbpath / "image_2"
    imageformat = "jpg"
    maskpath = dbpath / "mask"
    maskformat = "png"

    def im2mask(image):
        return maskpath / f"{image.stem}_mask.{maskformat}"

    def mask2im(mask):
        return imagepath / f"{mask.stem}.{imageformat}"

    def im2info(image):
        """
        Load info about image
        """
        dfs = {}
        for file in (dbpath / image.stem).glob("*.npy"):
            data = np.load(str(file), allow_pickle=True)
            name = file.stem.split("_")[-1]
            dfs[name] = pd.DataFrame(data)
        return dfs

    def rect_to_slice(rect_pts):
        """
        Convert cv-style rect to numpy-style slice
        """
        (x0, y0), (x1, y1) = rect_pts

        return (slice(y0, y1), slice(x0, x1))

    for image in imagepath.glob(f"*.{imageformat}"):

        # Image id
        image_i = image.stem

        im = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)

        t, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if args.threshold:
            im = thresh.copy()


        # Cut out legends
        roi = (slice(250, 4300), slice(375, 5630))
        im = im[roi]

        # Top right of frame (x,y)
        tr = np.array([375, 250])

        # Load info about image (bounding boxes etc.)
        data = im2info(image)

        # Group by symbol code
        for i, group in data["symbols"].groupby(2):
            symbol_code = group[2].iloc[0]
            symbol_path = outPath / str(symbol_code)

            symbol_path.mkdir(exist_ok=True)

            # Stack column 1 (rects)
            # Reshape to cv-style rects
            # Subtract top-right of legend sheet
            symbols = np.stack(group.iloc[:, 1]).reshape(-1, 2, 2) - tr

            # For each symbol rect
            for i, s in enumerate(symbols):
                # cut out the symbol from image
                symbol_image = im[rect_to_slice(s)]

                # Save with image id
                cv2.imwrite(str(symbol_path / f"im{image_i}_sym{i}.png"), symbol_image)
