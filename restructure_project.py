import os
import shutil

BASE_DIR = os.getcwd()

PROJECT1_DIR = os.path.join(BASE_DIR, "project1")

# Create project1 folder
os.makedirs(PROJECT1_DIR, exist_ok=True)

# Folders to move
folders_to_move = ["app", "src", "data"]

for folder in folders_to_move:
    src_path = os.path.join(BASE_DIR, folder)
    dest_path = os.path.join(PROJECT1_DIR, folder)

    if os.path.exists(src_path):
        print(f"Moving folder: {folder} → project1/")
        shutil.move(src_path, dest_path)
    else:
        print(f"Skipping {folder} (not found)")

# Files to move
files_to_move = ["requirements.txt", ".env"]

for file in files_to_move:
    src_file = os.path.join(BASE_DIR, file)
    dest_file = os.path.join(PROJECT1_DIR, file)

    if os.path.exists(src_file):
        print(f"Moving file: {file} → project1/")
        shutil.move(src_file, dest_file)
    else:
        print(f"Skipping {file} (not found)")

# Create empty project2 and project3 folders
for proj in ["project2", "project3"]:
    proj_path = os.path.join(BASE_DIR, proj)
    os.makedirs(proj_path, exist_ok=True)
    print(f"Created folder: {proj}/")

print("\n✅ Restructuring complete!")