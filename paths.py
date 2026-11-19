import os

# Define your project paths here (update these strings to match your actual folders!)
BASE_DIR = os.path.abspath(".")
MAP_DIR = os.path.join(BASE_DIR, "maps")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Automatically create these folders if they don't exist yet
for directory in [MAP_DIR, DATA_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
