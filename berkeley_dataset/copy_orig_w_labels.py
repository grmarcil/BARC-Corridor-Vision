import os
import skimage
import skimage.io

DATASET_DIRECTORY = "/Users/Greg/berkeley/CS280/project/berkeley_dataset"

def main():
    # Read all filenames from labels directory
    for filename in os.listdir(DATASET_DIRECTORY + "/labels"):
        if filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0]
            orig_path = DATASET_DIRECTORY + "/originals/" + base_name + ".jpg"
            save_path = DATASET_DIRECTORY + "/originals_with_labels/" + base_name + ".png"
            # Open corresponding original .jpg
            orig = skimage.io.imread(orig_path)
            # Convert to .png and save in /originals_with_labels
            skimage.io.imsave(save_path, orig)

if __name__ == '__main__':
    main()
