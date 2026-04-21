# Fredrick Farouk. footnotes4.py
# This fourth helper file uses the dataset created in footnotes3.py to structure the BUS-UCLM dataset.

import os
import shutil

# This translator was also printed in footnotes3.py
translator = {'ALWI': '1', 'ANAT': '2', 'ANFO': '3', 'ASSC': '4', 'CAWI': '5', 'CHCO': '6', 'CHSP': '7', 'CHVI': '8', 'CODE': '9', 'COPE': '10', 'COST': '11', 'COVA': '12', 'CRCI': '13', 'DAPA': '14', 'ELCO': '15', 'FLBA': '16', 'FLKA': '17', 'FUHI': '18', 'HESN': '19', 'HUBL': '20', 'KIFO': '21', 'LOTI': '22', 'MENE': '23', 'NIRO': '24', 'ORPE': '25', 'OSCU': '26', 'PAGY': '27', 'PLBA': '28', 'POFR': '29', 'RARE': '30', 'SECH': '31', 'SHST': '32', 'SIBA': '33', 'STSP': '34', 'TOCI': '35', 'UNCU': '36', 'VITR': '37', 'WAQU': '38'}
source_folder = "BUS-UCLM/Masks"

# Commented code here moves everything into correct numbers

for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    # Skip directories
    if not os.path.isfile(file_path):
        continue

    # Get first 4 letters (without extension)
    prefix = filename[:4]

    destination_folder_name = translator[prefix]
    destination_folder = os.path.join(source_folder, destination_folder_name)

    # Create folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move file
    shutil.move(file_path, os.path.join(destination_folder, filename))
    print(f"Moved {filename} → {destination_folder_name}")

import csv

source_directory = "BUS-UCLM/Masks"
csv_path = "BUS-UCLM/patient_level_information.csv"

folder_map = {}

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        translated_name = row["Patient"].strip()
        new_parent = row["Label"].strip()
        folder_map[translated_name] = new_parent

# Process folders
for folder_name in os.listdir(source_directory):
    folder_path = os.path.join(source_directory, folder_name)

    if not os.path.isdir(folder_path):
        continue

    if folder_name in folder_map:
        new_parent_name = folder_map[folder_name]
        new_parent_path = os.path.join(source_directory, new_parent_name)

        # Create parent folder if needed
        os.makedirs(new_parent_path, exist_ok=True)

        destination_path = os.path.join(new_parent_path, folder_name)

        shutil.move(folder_path, destination_path)

        print(f"Moved '{folder_name}' → '{new_parent_name}/'")
    else:
        print(f"Folder '{folder_name}' not found in CSV (unexpected)")