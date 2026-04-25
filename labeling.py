import os
import numpy as np
import msvcrt as ms
import subprocess
import pickle
def data_labeling(times: int, label: str):
    folder_path = f'data/raw/{label}' 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    saved = 0
    while saved < times:

        file_path = f"data/raw/{label}/sample_{saved}.npy"

        subprocess.run([
        "python",
        "main.py",
        "--mode", "run",
        "--recorder.file", file_path,
        ])
        print ("press escape to save the sample \n press q to quit")
        char = ms.getch()
        if char == b'\x1b': #esc button
            #saves the file
            saved+=1
        elif char == b'q':
            break
        else:
            os.remove(file_path)
            pass

def dataset_building(output_path):
    raw_folder = 'data/raw'
    dataset = {}
    for root, dirs, files in os.walk(raw_folder):
        label = os.path.basename(root)
        if label == "raw":
            continue
        for file in files:
         if file.endswith(".npy"):
            full_path = os.path.join(root,file)
            data = np.load(full_path)
            if label not in dataset:
                dataset[label] = []
            if data.shape != (20,2):
                continue
            if np.isnan(data).any() == True:
                continue
            dataset[label].append(data)
        
        
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print("Dataset saved to:", output_path)
            