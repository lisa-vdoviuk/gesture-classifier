# 🖐️ Gesture Recognition MPT

A real-time hand gesture recognition system utilizing MediaPipe and Hidden Markov Models (HMM) to classify dynamic alphabet trajectories.

Developed for the **Machine Perception** project.

---

## Overview

This project implements a modular pipeline in Python where live webcam data is processed into 3D hand landmarks, normalized, and fed into an HMM classifier. From raw camera input, the system can reliably predict the English alphabet in real time.

The project focuses on a signal-based architecture (`SignalHub`), rigorous data curation, and robust mathematical modeling.

---

## Alphabet Dataset

We recorded, cleaned, and trained the model to recognize the English alphabet.

<table align="center">
  <tr>
    <td align="center"><b>A</b><br><img width="120" src="https://github.com/user-attachments/assets/bd88ff3f-0a35-48e9-893a-da82689b0064" /></td>
    <td align="center"><b>B</b><br><img width="120" src="https://github.com/user-attachments/assets/89ddf4d5-10ac-45cb-bc47-147e96695ab1" /></td>
    <td align="center"><b>C</b><br><img width="120" src="https://github.com/user-attachments/assets/50fd1856-997b-4dbe-b6af-87f8dda4aa73" /></td>
    <td align="center"><b>D</b><br><i>WIP</i></td>
    <td align="center"><b>E</b><br><img width="120" src="https://github.com/user-attachments/assets/6b090f1b-1daa-4af3-a099-96bf9ffcf9e4" /></td>
    <td align="center"><b>F</b><br><img width="120" src="https://github.com/user-attachments/assets/de2228e2-3d63-4b11-a09e-1f77a1ad0952" /></td>
    <td align="center"><b>G</b><br><img width="120" src="https://github.com/user-attachments/assets/c97937a4-48e7-4301-9393-28350b8e4460" /></td>
  </tr>
  <tr>
    <td align="center"><b>H</b><br><img width="120" src="https://github.com/user-attachments/assets/c922bfd1-7048-411d-a339-274d02ec2d9b" /></td>
    <td align="center"><b>I</b><br><img width="120" src="https://github.com/user-attachments/assets/aa5be910-cb58-417d-98ee-b6d55f796ca3" /></td>
    <td align="center"><b>J</b><br><img width="120" src="https://github.com/user-attachments/assets/95d631fb-8a21-4aaa-bfa7-f2a188d63613" /></td>
    <td align="center"><b>K</b><br><img width="120" src="https://github.com/user-attachments/assets/c17eb66a-8ab6-4278-a7c2-2244f76880bd" /></td>
    <td align="center"><b>L</b><br><img width="120" src="https://github.com/user-attachments/assets/1607ec33-396d-4d0d-89de-b6b7fecfe278" /></td>
    <td align="center"><b>M</b><br><img width="120" src="https://github.com/user-attachments/assets/9a0337eb-3e09-4226-9b32-42e3b42c9bb3" /></td>
    <td align="center"><b>N</b><br><img width="120" src="https://github.com/user-attachments/assets/4a104590-bb71-4b17-9213-41e88128e642" /></td>
  </tr>
  <tr>
    <td align="center"><b>O</b><br><img width="120" src="https://github.com/user-attachments/assets/96db3aac-d15e-43df-b159-e4b8bf0c673c" /></td>
    <td align="center"><b>P</b><br><img width="120" src="https://github.com/user-attachments/assets/a1962dfc-cca5-4b07-ab67-5fbb6671343e" /></td>
    <td align="center"><b>Q</b><br><img width="120" src="https://github.com/user-attachments/assets/c3fa530b-3442-4c6b-9035-26a37384ea29" /></td>
    <td align="center"><b>R</b><br><img width="120" src="https://github.com/user-attachments/assets/b5b16574-dc4f-4596-a21e-641c7e3b64a6" /></td>
    <td align="center"><b>S</b><br><img width="120" src="https://github.com/user-attachments/assets/795a0688-fd7c-4faa-a86a-f5aec94843f8" /></td>
    <td align="center"><b>T</b><br><img width="120" src="https://github.com/user-attachments/assets/28e5c1b6-5f44-4b84-b37a-918d8b16df40" /></td>
    <td align="center"><b>U</b><br><img width="120" src="https://github.com/user-attachments/assets/ac268ab6-2d27-44cb-9236-c671765af306" /></td>
  </tr>
  <tr>
    <td align="center"><b>V</b><br><img width="120" src="https://github.com/user-attachments/assets/05e3ada6-cf81-44f6-bde6-87ad3cb16067" /></td>
    <td align="center"><b>W</b><br><img width="120" src="https://github.com/user-attachments/assets/1a4256df-7a93-4797-90b3-32e6196a92b6" /></td>
    <td align="center"><b>X</b><br><img width="120" src="https://github.com/user-attachments/assets/0c9579a6-7d05-46cb-ae54-fb5905a1880e" /></td>
    <td align="center"><b>Y</b><br><img width="120" src="https://github.com/user-attachments/assets/d24b421e-ee30-4154-b14b-3337ebfd39d9" /></td>
    <td align="center"><b>Z</b><br><img width="120" src="https://github.com/user-attachments/assets/90107ce6-8501-43fd-a9ee-1595748a029e" /></td>
    <td></td>
    <td></td>
  </tr>
</table>

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/jaboll-ai/GestureRecognitionMPT.git
cd GestureRecognitionMPT
pip install -r requirements.txt
```

*(Note: MediaPipe `.task` models will automatically download the first time you run the application).*

---

## Running the Application

```bash
python main.py --mode run
```

If you are using an external webcam, open `config.yml` and change `deviceindex: 0` to your specific camera index (e.g., `1` or `2`).

---

## Controls & Usage

The recording system is controlled entirely via physical hand gestures:

| Input | Action |
|---|---|
| ☝️ **Index Finger Up** | Start recording trajectory |
| ✊ **Closed Fist** | Stop recording and save sample |
| **Top Menu** | Switch between Training / Dataset Build / Classification |

### 1. Recording Data (Training Mode)
1. In the live window, use the top menu: **Training -> Start training mode**.
2. Enter the letter you want to record and the target number of samples.
3. Point your index finger UP (`☝️`) to start. Draw the letter. Close your hand into a FIST (`✊`) to stop and save the sample.
4. Repeat until you reach the target count.

### 2. Building & Training
1. Once all letters are recorded and QA'd, go to **Training -> Build dataset**. This generates the `data/dataset.pkl` file.
2. Click **Training -> Train HMM model**. This will extract features, run the grid search, evaluate the accuracy, and save the model to `data/hmm_classifier.pkl`.

### 3. Live Classification
Once the model is trained, the system will default to Classification Mode. Simply draw a letter in front of the camera, and the HMM prediction (with confidence score) will appear on the screen!

---

## Data Engineering & QA

Training a Hidden Markov Model requires clean state transitions. We utilized a two-step data pipeline:

1. **Collection:** Raw recordings are saved to `data/raw/`.
2. **Quality Assurance:** We manually evaluate trajectories using `visualization.py`. Samples with tracking glitches (teleporting points) or incorrect stroke starts are discarded.
3. **Compilation:** Verified samples are moved to `data/prepared/`, where the final `dataset.pkl` is built. We aim for ~30 clean samples per class to prevent model bias.

---

## Project Structure

```
GestureRecognitionMPT/
├── data/
│   ├── raw/                 # Unfiltered gesture recordings
│   ├── prepared/            # QA-verified golden samples
│   ├── dataset.pkl          # Compiled dataset for training
│   └── hmm_classifier.pkl   # Trained HMM model weights
├── GestureRecognition/
│   ├── modules/             # SignalHub application modules
│   └── hmmclassifier.py     # HMM math and grid-search logic
├── main.py                  # Entry point and engine setup
├── config.yml               # Central configuration parameters
├── visualization.py         # Matplotlib trajectory evaluation
├── requirements.txt
```

---

## Module Overview

- **`HandDetector`**: Analyzes the webcam frame using MediaPipe and extracts 21 3D hand landmarks.
- **`GestureState`**: Identifies specific static triggers (`Pointing_Up` to start, `Closed_Fist` to stop).
- **`Preprocessor`**: Tracks the index fingertip over time, centers/scales the coordinate trajectory, and removes jitter.
- **`TrailMarker`**: Provides real-time visual feedback to the user by drawing a fading cyan trail using a deque memory and affine mapping.
- **`TrainingController`**: A PyQt5 UI wrapper managing the application state, saving `.npy` files, and initiating the offline HMM training process.
- **`HMMClassifier` / `HiddenMarkov`**: Resamples sequences to 40 points, runs a Grid Search to find optimal parameters, and takes live preprocessed trajectories to predict the active gesture.

---

## Team

- Yelyzaveta Vdoviuk
- Oleksii Zvirkovskyi
- Emen Fouda
- Sofiene Bembli

---
## License

For academic use only.
