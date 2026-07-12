# Gesture Classifier MPT

<p align="center">
  <img width="32%" src="https://github.com/user-attachments/assets/3400fb6d-aba7-4841-b91a-36e83ed5aab0" />
  <img width="32%" src="https://github.com/user-attachments/assets/4195b148-cad9-4004-b595-3e78123519aa" />
  <img width="32%" src="https://github.com/user-attachments/assets/34a5e846-0bdd-4b67-ba3d-d31fdaa7d948" />
</p>

A real-time hand gesture classifier system utilizing MediaPipe and Hidden Markov Models (HMM) to classify dynamic alphabet trajectories.

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
    <td align="center"><b>D</b><br><img width="120" src="https://github.com/user-attachments/assets/ffccba82-b909-422a-af06-c7ae2a74a44d" /></td>
    <td align="center"><b>E</b><br><img width="120" src="https://github.com/user-attachments/assets/6b090f1b-1daa-4af3-a099-96bf9ffcf9e4" /></td>
    <td align="center"><b>F</b><br><img width="120" src="https://github.com/user-attachments/assets/de2228e2-3d63-4b11-a09e-1f77a1ad0952" /></td>
    <td align="center"><b>G</b><br><img width="120" src="https://github.com/user-attachments/assets/c97937a4-48e7-4301-9393-28350b8e4460" /></td>
  </tr>
  <tr>
    <td align="center"><b>H</b><br><img width="120" src="https://github.com/user-attachments/assets/7de7906a-53dc-4c1d-93e0-15646c5eacc1" /></td>
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
git clone https://github.com/lisa-vdoviuk/gesture-classifier.git
cd gesture-classifier
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

The recording system is controlled via physical hand gestures and a PyQt5 menu overlay:

| Input | Action |
| --- | --- |
| ☝️ **Index Finger Up** | Start recording trajectory |
| ✊ **Closed Fist** | Stop recording and save sample |
| **`Ctrl + S`** | Shortcut to manually save current sample |
| **Top Menu** | Switch between Training / Dataset Build / Classification |

### 1. Recording Data (Training Mode)

1. In the live window, use the top menu: **Training -> Start training mode**.
2. Enter the letter you want to record and the target number of samples.
3. Point your index finger UP (`☝️`) to start drawing the letter. Close your hand into a FIST (`✊`) or press `Ctrl+S` to stop and save the sample.
4. Repeat until you reach the target count.

### 2. Building & Training

1. Once all letters are recorded and QA'd, go to **Training -> Build dataset**. This compiles the records into the `data/dataset.pkl` file.
2. Click **Training -> Train HMM model**. This extracts features, splits data for offline validation, runs the grid search, evaluates accuracy, and saves the weights to `data/hmm_classifier.pkl`.

### 3. Live Classification

Once the model is trained, the system defaults to Classification Mode. Simply draw a letter in front of the camera, and the HMM prediction (with confidence score) will appear on the screen!

---

## System Architecture

The project is built on the `SignalHub` framework, passing data through a pipeline of isolated modules:

* **`HandDetector`**: Analyzes the webcam frame using MediaPipe and extracts 21 3D hand landmarks.
* **`GestureState`**: Identifies specific static triggers (`Pointing_Up` to start, `Closed_Fist` to stop).
* **`Preprocessor`**: Tracks the index fingertip over time, centers/scales the coordinate trajectory, and prepares the matrix.
* **`TrailMarker`**: Provides real-time visual feedback to the user by drawing a fading cyan trail using a deque memory and affine mapping.
* **`TrainingController`**: A PyQt5 UI wrapper managing the application state, saving `.npy` files, and initiating the offline HMM training process.
* **`HMMClassifier` / `HiddenMarkov`**: Resamples sequences, runs a Grid Search to find optimal parameters, and takes live preprocessed trajectories to predict the active gesture.

---

## Design Decisions & Optimizations

To ensure high accuracy across different users and drawing speeds, we implemented several specific design choices in our data engineering and ML pipeline:

### 1. Preprocessor Normalization

Raw coordinates depend on where the user stands in the frame. The preprocessor achieves **spatial invariance** by:

* **Mean Centering:** Calculating the spatial mean center of the captured index-finger trajectory and subtracting it from all points.
* **Bounding Box Scaling:** Normalizing the entire array by dividing by the maximum absolute value, ensuring all gestures scale within a `[-1, 1]` footprint.
* **End-Trimming:** Truncating the final 5 frames (`arr[:-5]`) of the trajectory sequence to eliminate tracking distortion caused when transitioning into the trailing closed-fist gesture.

### 2. HMM Feature Extraction & Resampling

Because users draw at different speeds, raw sequences have vastly different frame counts.

* **Linear Resampling:** All sequences are strictly interpolated to exactly 40 points (`resample_len=40`). This standardizes the time-series length across all character variants.
* **Velocity Features (`xy_dxy`):** Instead of just looking at static points (`x, y`), the model calculates the delta/velocity between consecutive frames (`dxy`). This provides the HMM with crucial context about the *direction* of the stroke.

### 3. Grid Search & Regularization

During training, the system runs an automated Grid Search with a 25% validation split. It tests hidden states (`n_components` from 2 up to a maximum of 5, as configured by the controller) to find the optimal complexity for each specific letter. We also implemented covariance regularization (`min_covar_options`) to prevent matrix singularities when a gesture path has minor spatial variance.

### 4. Confidence Score Calculation

Gaussian HMMs output unnormalized log-likelihoods, which are difficult to interpret as absolute metrics. We calculate a readable, relative posterior probability distribution (0% - 100%) by:

* **Prior Normalization:** Whole-sequence log-priors are matched to a per-point scale using the resample length used during scoring.
* **Posterior Mapping:** Combining likelihoods and adjusted log-priors, shifting by the maximum score row entry for numerical stability, and utilizing exponential normalization to yield relative probabilities across the trained classes.

---

## Data Engineering & QA

Training a Hidden Markov Model requires clean state transitions. We utilized a structured data pipeline:

1. **Collection:** Raw recordings are saved as `.npy` array paths grouped under directories inside `data/raw/`.
2. **Quality Assurance:** We manually evaluate trajectories using `visualization.py`. Samples with tracking glitches (teleporting points) or incorrect stroke starts are discarded.
3. **Compilation:** The final `dataset.pkl` is built directly from valid `data/raw/` entries. We use 30 clean samples per class to prevent model bias.

---

## Interpretation of Results

*Note: This section summarizes the final accuracy and confusion matrix generated during the offline training phase.*

* **Overall Accuracy:** [Final accuracy here]
* **Observations:** [Final confusion matrix observation here]

---

## Project Structure

```
gesture-classifier/
├── data/
│   ├── raw/                 # Unfiltered gesture recordings (.npy format)
│   ├── dataset.pkl          # Compiled dataset for training
│   └── hmm_classifier.pkl   # Trained HMM model weights
├── GestureRecognition/
│   ├── modules/             # SignalHub application modules
│   │   ├── gesturestate.py
│   │   ├── handdetector.py
│   │   ├── hiddenmarkov.py
│   │   ├── preprocessor.py
│   │   ├── recorder.py
│   │   ├── trailmarker.py
│   │   └── trainingcontroller.py
│   └── hmmclassifier.py     # HMM math and grid-search logic
├── main.py                  # Entry point and engine setup
├── config.yml               # Central configuration parameters
├── visualization.py         # Matplotlib trajectory evaluation
└── requirements.txt         # Project dependencies
```

---

## Team

* Yelyzaveta Vdoviuk
* Oleksii Zvirkovskyi
* Emen Fouda
* Sofiene Bembli

---

## License

For academic use only.
