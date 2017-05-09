import os
import numpy as np
import skimage
import skimage.io
import xml.etree.ElementTree as ET
import cv2

DATASET_DIRECTORY = "/Users/Greg/berkeley/CS280/project/berkeley_dataset"


def process_image(xml_filename):
    # Construct file paths
    # Inputs
    base_name = os.path.splitext(xml_filename)[0]
    xml_file_path = DATASET_DIRECTORY + "/xml/" + xml_filename
    orig_file_path = DATASET_DIRECTORY + "/originals/" + base_name + ".jpg"
    # Outputs
    label_path = DATASET_DIRECTORY + "/labels/" + base_name + ".png"
    bw_path = DATASET_DIRECTORY + "/bw_labels/" + base_name + ".png"
    overlay_path = DATASET_DIRECTORY + "/overlaid_labels/" + base_name + ".png"

    # Initialize output images and load original
    label_output = np.zeros([240, 320], np.uint8)
    bw_output = np.zeros([240, 320], np.uint8)
    overlay_output = skimage.io.imread(orig_file_path)
    original = skimage.io.imread(orig_file_path)

    # Load and parse hand-labeled XML from LabelMe tool
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    objects = root.findall('object')

    # Extract list of OpenCV contours (which are just lists of [x,y] points)
    # from all found polygon objects in the XML
    contours, is_labeled = extract_contours(objects)

    if is_labeled:
        # drawContours params:
        # (image, contour array, which contour(negative=all), color,
        # thickness(negative=fill))

        # Build label and bw images
        cv2.drawContours(label_output, contours, -1, (1), -1)
        cv2.drawContours(bw_output, contours, -1, (255), -1)

        # Build overlay image: original image with green shadow over floor
        alpha = 0.5
        cv2.drawContours(overlay_output, contours, -1, (50, 205, 50), -1)
        cv2.addWeighted(overlay_output, alpha, original, 1 - alpha, 0,
                        overlay_output)

        # Save outputs
        skimage.io.imsave(label_path, label_output)
        skimage.io.imsave(bw_path, bw_output)
        skimage.io.imsave(overlay_path, overlay_output)


def extract_contours(objects):
    contours = []
    is_labeled = False

    for obj in objects:
        polygons = obj.findall('polygon')
        for poly in polygons:
            # If polygons found, flip is_labeled flag and extract [x,y] pairs
            # into a list
            is_labeled = True

            contour = np.array([])
            pts = poly.findall('pt')
            for pt in pts:
                x = int(pt.findall('x')[0].text)
                y = int(pt.findall('y')[0].text)
                contour = np.append(contour, [x, y])
            # Stack points into 2d array expected by cv2.drawContours
            contour = np.reshape(contour, (len(pts), 2)).reshape(
                (-1, 1, 2)).astype(np.int32)
            contours.append(contour)
    return contours, is_labeled


def main():
    for filename in os.listdir(DATASET_DIRECTORY + "/xml"):
        if filename.endswith(".xml"):
            process_image(filename)


if __name__ == '__main__':
    main()
