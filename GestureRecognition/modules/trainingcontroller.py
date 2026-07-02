import os
import time
import pickle
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import QAction, QInputDialog, QMessageBox
from SignalHub import Module, GALY
import mediapipe as mp



class TrainingController(Module):
    def __init__(
        self,
        outputSignal="app_state",
        raw_dir="data/raw",
        dataset_path="data/dataset.pkl",
        model_path="data/hmm_classifier.pkl",
    ):
        super().__init__(
            inputSignals=["config", "preprocessor", "gesture_state"],
            outputSchema={
                "type": "object",
                "properties": {
                    outputSignal: {},
                },
            },
            name="training_controller",
        )

        self.outputSignal = outputSignal

        self.raw_dir = Path(raw_dir)
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)

        self.mode = "classification"

        self.current_label = None
        self.target_samples = 0
        self.saved_samples = 0

        self.last_trajectory = None

        self.model_version = 0
        self.ui_initialized = False

        self.is_recording = False

        self.current_gesture = "None"
        self.status_message = "WAIT"

    def start(self, data):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            self.outputSignal: self._state(),
        }

    def step(self, data):
        self._init_ui_once()

        gesture_state = data.get("gesture_state")
        trajectory = data.get("preprocessor")

        gesture = gesture_state.get("gesture", "None")
        score = gesture_state.get("score", 0.0)
        is_recording = gesture_state.get("is_recording", False)
        recording_stopped = gesture_state.get("recording_stopped", False)

        self.is_recording = is_recording
        self.current_gesture = f"{gesture} {score:.2f}"

        if is_recording:
            self.status_message = "REC"
        elif not is_recording and self.status_message == "REC":
            self.status_message = "WAIT"

        if recording_stopped:
            if self._is_valid_trajectory(trajectory):
                self.last_trajectory = trajectory.copy()
                if self.mode == "training":
                    self._auto_save_sample()
                    self.status_message = f"SAVED {self.saved_samples}/{self.target_samples}"
                else:
                    self.status_message = "STOPPED"
            else:
                self.status_message = "INVALID TRAJECTORY"
            
        return {
            self.outputSignal: self._state(),
            "galy": self._make_overlay(),
            }

    def stop(self, data):
        return {}

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _init_ui_once(self):
        if self.ui_initialized:
            return

        try:
            import SignalHub.galyQT as galy_qt
            window = galy_qt.window
        except Exception:
            return

        if window is None:
            return

        menu = window.menuBar().addMenu("Training")

        start_action = QAction("Start training mode", window)
        start_action.triggered.connect(self._start_training_mode)
        menu.addAction(start_action)

        save_action = QAction("Save current sample", window)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_current_sample)
        menu.addAction(save_action)

        build_action = QAction("Build dataset", window)
        build_action.triggered.connect(self._build_dataset_action)
        menu.addAction(build_action)

        train_action = QAction("Train HMM model", window)
        train_action.triggered.connect(self._train_hmm_model)
        menu.addAction(train_action)

        classify_action = QAction("Switch to classification mode", window)
        classify_action.triggered.connect(self._switch_to_classification_mode)
        menu.addAction(classify_action)

        self.ui_initialized = True

    def _start_training_mode(self):
        label, ok = QInputDialog.getText(
            None,
            "Training mode",
            "Gesture label:",
        )

        if not ok:
            return

        label = label.strip()

        if not label:
            QMessageBox.warning(None, "Training mode", "Label cannot be empty.")
            return

        count, ok = QInputDialog.getInt(
            None,
            "Training mode",
            "Number of samples:",
            value=10,
            min=1,
            max=10000,
        )

        if not ok:
            return

        self.mode = "training"
        self.current_label = label
        self.target_samples = count
        self.saved_samples = 0

        label_dir = self.raw_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        QMessageBox.information(
            None,
            "Training mode",
            (
                f"Training mode started.\n\n"
                f"Label: {label}\n"
                f"Samples: {count}\n\n"
                f"Show gesture and press Ctrl+S to save each sample."
            ),
        )

    def _save_current_sample(self):
        if self.mode != "training":
            QMessageBox.warning(
                None,
                "Training mode",
                "You are not in training mode.",
            )
            return

        if self.current_label is None:
            QMessageBox.warning(
                None,
                "Training mode",
                "No label selected.",
            )
            return

        if self.last_trajectory is None:
            QMessageBox.warning(
                None,
                "Training mode",
                "No valid trajectory yet. Keep your hand visible.",
            )
            return

        if self.saved_samples >= self.target_samples:
            QMessageBox.information(
                None,
                "Training mode",
                "Target number of samples already collected.",
            )
            return

        file_path = self._next_sample_path(self.current_label)
        np.save(file_path, self.last_trajectory)

        self.saved_samples += 1
        if self._finish_training_if_needed():
            QMessageBox.information(
                None,
                "Training complete",
                "Target number of samples collected. Training complete.",
            )

    def _auto_save_sample(self):
        if self.current_label is None or self.saved_samples >= self.target_samples:
            return
        file_path = self._next_sample_path(self.current_label)
        np.save(file_path, self.last_trajectory)
        self.saved_samples += 1
        self._finish_training_if_needed()

    def _train_hmm_model(self):
        try:
            self._run_offline_hmm_training()
        except Exception as e:
            QMessageBox.critical(
                None,
                "HMM training failed",
                str(e),
            )
            return

        self.model_version += 1
        self.mode = "classification"

        QMessageBox.information(
            None,
            "HMM training",
            (
                "HMM model trained successfully.\n\n"
                "Switched back to classification mode."
            ),
        )

    def _switch_to_classification_mode(self):
        self.mode = "classification"

        QMessageBox.information(
            None,
            "Classification mode",
            "Switched to classification mode.",
        )

    # ------------------------------------------------------------
    # Training bridge
    # ------------------------------------------------------------

    def _run_offline_hmm_training(self):
        from GestureRecognition.hmmclassifier import HMMClassifier
        dataset = self._build_dataset()
        if not dataset:
            try:
                return self._build_dataset_action()
            except Exception as e:
                QMessageBox.critical(
                    None,
                    "Dataset build failed",
                    str(e),
                )
                return
        

        classifier = HMMClassifier(n_components=5)
        train_data, test_data = classifier.train_test_split_dataset(dataset=dataset, test_size=0.2)
        classifier.fit(train_data)
        classifier.save(self.model_path)
        print(f"--- Results of testing ---")
        classifier.evaluate_classifier(test_data)



    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _is_valid_trajectory(self, trajectory):
        if trajectory is None:
            return False

        if not isinstance(trajectory, np.ndarray):
            return False

        if trajectory.shape[0] < 20 or trajectory.shape[1] != 2:
            return False

        if np.isnan(trajectory).any():
            return False

        return True

    def _finish_training_if_needed(self):
        if self.mode == "training" and self.saved_samples >= self.target_samples:
            self.mode = "classification"
            self.is_recording = False
            self.status_message = "TRAINING COMPLETE"
            self._freeze_until = time.time() + 2.0
            self._build_dataset()
            return True
        return False
    
    def _build_dataset(self):
        dataset = {}
        for root, _, files in os.walk(self.raw_dir):
            label = os.path.basename(root)
            if label == self.raw_dir.name:
                continue
            for file in files:
                if not file.endswith(".npy"):
                    continue
                full_path = os.path.join(root, file)
                data = np.load(full_path)
                if data.shape[0] < 20 or data.shape[1] != 2:
                    continue
                if np.isnan(data).any():
                    continue
                if label not in dataset:
                    dataset[label] = []
                dataset[label].append(data)

        with open(self.dataset_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Dataset saved to: {self.dataset_path}")
        return dataset

    def _build_dataset_action(self):
        try:
            dataset = self._build_dataset()
            summary = "\n".join(f"{k}: {len(v)} samples" for k, v in dataset.items())
            QMessageBox.information(
                None,
                "Dataset built",
                f"Saved to {self.dataset_path}\n\n{summary or 'No samples found.'}",
            )
        except Exception as e:
            QMessageBox.critical(None, "Dataset build failed", str(e))

    def _next_sample_path(self, label):
        label_dir = self.raw_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        existing_indices = []

        for path in label_dir.glob("sample_*.npy"):
            try:
                index = int(path.stem.replace("sample_", ""))
                existing_indices.append(index)
            except ValueError:
                pass

        next_index = max(existing_indices, default=-1) + 1

        return label_dir / f"sample_{next_index}.npy"

    def _state(self):
        return {
            "mode": self.mode,
            "label": self.current_label,
            "saved_samples": self.saved_samples,
            "target_samples": self.target_samples,
            "model_version": self.model_version,
            "model_path": str(self.model_path),
        }

    def _make_overlay(self):
        galy = GALY()
        galy.layer("training-controller", alwaysVisible=True)

        if self.mode == "training":
            rec_state = "REC" if self.is_recording else "WAIT"
            label = self.current_label if self.current_label else "-"
            text = (
                f"{rec_state} | {label} | "
                f"{self.saved_samples}/{self.target_samples} | "
                f"{self.status_message}"
            )
        else:
            text = f"CLASSIFY | {self.current_gesture}"

        galy.putText(
            text=text,
            org=(10, 18),
            fontScale=0.35,
            color=(255, 255, 255),
            thickness=1,
        )

        return galy
