import os
import numpy as np
import skimage
import skimage.io
import xml.etree.ElementTree as ET
import cv2

DATASET_DIRECTORY = "/Users/Greg/berkeley/CS280/project/berkeley_dataset"

def process_image(output_filename):
    # Construct file paths
    # Inputs
    base_name = os.path.splitext(output_filename)[0]
    label_file_path = DATASET_DIRECTORY + "/output/bw_output/" + output_filename
    original_file_path = DATASET_DIRECTORY + "/originals/" + base_name + ".jpg"
    # Outputs
    overlay_path = DATASET_DIRECTORY + "/output/overlay_output/" + base_name + ".png"

    # Initialize output images and load CNN output
    label = skimage.io.imread(label_file_path)
    original = skimage.io.imread(original_file_path)
    overlay_output = original.copy()

    # Build overlay image
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            if label[x][y] == 255:
                overlay_output[x][y] = np.array([50, 205, 50])

    # Blend overlay image
    alpha = 0.5
    cv2.addWeighted(overlay_output, alpha, original, 1 - alpha, 0,
                    overlay_output)

    # Save outputs
    skimage.io.imsave(overlay_path, overlay_output)


def main():
    for filename in os.listdir(DATASET_DIRECTORY + "/output/bw_output/"):
        if filename.endswith(".png"):
            process_image(filename)


if __name__ == '__main__':
    main()
