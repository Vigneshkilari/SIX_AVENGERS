import os
import numpy as np
def load_data_from_npy():
    data_dir = "C:/_Vignesh_N/1_six_avengers_final/data_np"
    img_path = os.path.join(data_dir, 'img.npy')
    map_path = os.path.join(data_dir, 'map.npy')

    print(f"Attempting to load from directory: {data_dir}")
    print(f"Image path: {img_path}")
    print(f"Map path: {map_path}")

    try:
        img = np.load(img_path)
        map_data = np.load(map_path)
        return img, map_data
    except FileNotFoundError:
        print(f"Error: 'img.npy' or 'map.npy' not found in {data_dir}")
        return None, None  # Return None to indicate failure
    except Exception as e:  # Catch other potential errors
        print(f"An error occurred: {e}")
        return None, None
