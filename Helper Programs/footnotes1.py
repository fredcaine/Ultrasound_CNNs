# Fredrick Farouk. footnotes.py
# This file and footnotes2.py generally contains programs used for formatting, or restructuring datasets.

# Originally, this file structures the dataset by moving the images into an images folder.
# Now, it moves the images into the masks folder.
import os
import shutil

source_root = "Images"
target_root = "Masks"

for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.endswith("_mask.png"):
            source_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, source_root)

            target_folder = os.path.join(target_root, relative_path)
            os.makedirs(target_folder, exist_ok=True)

            target_path = os.path.join(target_folder, file)

            shutil.move(source_path, target_path)

print("Done.")