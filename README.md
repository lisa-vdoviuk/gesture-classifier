# Gesture Classifier MPT

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmark%20Tracking-00C853)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8)
![PyQt5](https://img.shields.io/badge/UI-PyQt5-blueviolet)
![HMM](https://img.shields.io/badge/Model-Gaussian%20HMM-darkgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E)
![Metrics](https://img.shields.io/badge/Evaluation-Confidence%20%26%20Likelihood-orange)
![License](https://img.shields.io/badge/License-Academic%20Use%20Only-red)

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
3. Point your index finger UP (`☝️`) to start drawing the letter. Close your hand into a FIST (`✊`) to stop and save the sample.
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

### 4. Live Prediction & Confidence Score Calculation

Gaussian HMMs natively output unnormalized log-likelihoods, which are highly sensitive to sequence length and difficult to interpret as absolute absolute metrics. To resolve this in real-time mode, the system normalizes the outputs per point, injects training priors, and converts the results into readable probabilities ($0\% - 100\%$).

#### The Scoring Formula
For each gesture class, the live classification score is calculated using the following joint per-point formulation:

$$\text{Score} = \left(\frac{\text{Log-Likelihood}}{N}\right) + \text{Scaled Prior}$$

Where the variables represent:
*   **Log-Likelihood:** The raw HMM shape-matching score reflecting how well the drawn trajectory conforms to the structural states of that specific letter.
*   **$N$ (Trajectory Length):** The number of points in the sequence (standardized via linear interpolation to `resample_len = 40`). Dividing by $N$ normalizes the score, preventing longer gestures from being unfairly penalized compared to shorter ones.
*   **Scaled Prior:** A frequency bonus based on how many clean samples of this class exist in the training data. The whole-sequence log-prior is mathematically downscaled to the per-point level to maintain balance with the geometric shape score.

#### Prediction & Confidence Mapping
Once the scoring row is computed for all 26 classes, the final classification decision is made:

*   **Prediction ($\text{argmax}$):** The system selects the alphabet class that yields the highest combined score.
*   **Confidence Score ($\text{Softmax}$):** To convert raw exponential log-scales into intuitive probabilities, the scores are passed through a numerically stable **Softmax** function (shifted by the maximum score entry to avoid underflow/overflow errors). This maps the top prediction directly into a clear $0.0 - 1.0$ confidence metric displayed live on the screen.

---

## Data Engineering & QA

Training a Hidden Markov Model requires clean state transitions. We utilized a structured data pipeline:

1. **Collection:** Raw recordings are saved as `.npy` array paths grouped under directories inside `data/raw/`.
2. **Quality Assurance:** We manually evaluate trajectories using `visualization.py`. Samples with tracking glitches (teleporting points) or incorrect stroke starts are discarded.
3. **Compilation:** The final `dataset.pkl` is built directly from valid `data/raw/` entries. We use 30 clean samples per class to prevent model bias.

---

## Model Performance

Trained on the full alphabet (26 classes, 30 samples per class), evaluated on a held-out 20% test split.

**Overall accuracy: 96.8%**

### Key Metrics

<p align="center">
  <img width="85%" src="https://github.com/user-attachments/assets/6e255dcd-892c-4036-bdce-17e822a6992f" alt="Per-class accuracy" />
  <br/><sub>Per-class accuracy vs. overall average</sub>
</p>

<p align="center">
  <img width="65%" src="https://github.com/user-attachments/assets/6177738c-20a9-406a-b908-aa9004a838f4" alt="Normalized confusion matrix" />
  <br/><sub>Confusion matrix, normalized by true class (= per-class recall)</sub>
</p>

---

### Deep Dive: Detailed Metrics & Confidence

<details>
<summary><b>Click to expand Precision/Recall/F1 and Confidence Distribution</b></summary>
<br/>

**Precision / Recall / F1 Score per Class**
<p align="center">
  <img width="50%" src="https://github.com/user-attachments/assets/1088872e-9be4-445f-b677-7ce9c37c7102" alt="Precision / Recall / F1 per class" />
</p>

**Model Confidence per Predicted Class**
<p align="center">
  <img width="85%" src="https://github.com/user-attachments/assets/61edc326-aaee-4194-9e27-1a4ce67d7168" alt="Confidence distribution by predicted class" />
</p>
<i>Confidence spread varies noticeably by letter: short/simple strokes (<code>K</code>, <code>R</code>) get lower, noisier confidence, while distinctive multi-directional strokes (<code>U</code>, <code>I</code>) are classified with consistently high confidence.</i>

</details>

---

### Known Challenges

Test errors cluster around letters whose drawn trajectory is visually similar as a single continuous stroke:

| True label | Predicted as |
|:---:|:---:|
| **B** | A |
| **C** | G |
| **G** | B |
| **N** | D |
| **P** | B |

> **Note:** `B`, `G`, and `P` in particular share a similar "loop + vertical stroke" motion when drawn quickly — this is the main source of confusion for the HMM.


---

## Project Structure

```
gesture-classifier/
├── data/
│   ├── raw/                 # Gesture recordings (.npy format)
│   ├── dataset.pkl          # Compiled dataset for training
│   └── hmm_classifier.pkl   # Trained HMM model weights
├── GestureRecognition/
│   ├── modules/             # SignalHub application modules
│   │   ├── gesturestate.py
│   │   ├── handdetector.py
│   │   ├── hiddenmarkov.py
│   │   ├── preprocessor.py
│   │   ├── trailmarker.py
│   │   └── trainingcontroller.py
│   └── hmmclassifier.py     # HMM math and grid-search logic
├── main.py                  # Entry point and engine setup
├── config.yml               # Central configuration parameters
├── visualization.py         # Matplotlib trajectory evaluation
└── requirements.txt         # Project dependencies
```
## Possible Enhancements

- **Data augmentation** — synthetic jitter, rotation, and time-warping on existing trajectories to reduce reliance on collecting more raw samples per class.
- **Confusable-pair features** — add stroke start-point / direction-of-travel as an explicit feature to help disambiguate visually similar strokes (`B`/`G`/`P`).
- **Writer-independent validation** — split train/test by *person*, not just by sample, to test generalization across different users/hand sizes.
- **Adaptive resampling** — replace the fixed `resample_len=40` with DTW-based alignment instead of linear interpolation (chart below shows the problem of distribution).
<details>
  
**Trajectory Length Distribution**

<p align="center">
  <img width="50%" src="https://github.com/user-attachments/assets/92551ecf-c097-408f-9d24-0e11613fddf7" alt="Precision / Recall / F1 per class" />
</p>
</details>

---

## Team

* Yelyzaveta Vdoviuk
* Oleksii Zvirkovskyi
* Emen Fouda
* Sofiene Bembli

---

## License

For academic use only.
