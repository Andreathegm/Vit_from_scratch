import os
import random

def random_element_from_subfolders(folder_path: str,seed =None):
    if seed is not None:
        random.seed(seed)
        
    subfolders = [
        os.path.join(folder_path, d)
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    if not subfolders:
        raise ValueError(f"folder {folder_path} not contains subfolders")

    chosen_subfolder = random.choice(subfolders)

    elements = os.listdir(chosen_subfolder)
    if not elements:
        raise ValueError(f"Subfolder  '{chosen_subfolder}' is empty.")

    return os.path.join(chosen_subfolder, random.choice(elements))
