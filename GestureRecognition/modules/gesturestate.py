import time
from pathlib import Path
import numpy as np
from SignalHub import Module
import mediapipe as mp
import urllib.request
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureReqognizer = mp.tasks.vision.GestureRecognizer
GestureReqognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class GestureState(Module):
    def __init__(self, outputSignal="gesture_state"):
        super().__init__(
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="gesture_state",
        )
        self.outputSignal = outputSignal

        self.current_gesture = "None"
        self.is_recording = False

    def start(self, data):
        gesture_model_path = Path("gesture_recognizer.task")

        if not gesture_model_path.exists():
            url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            print(f"Downloading model to {gesture_model_path}...")
            urllib.request.urlretrieve(url, gesture_model_path)
            print("Done.")
        
        options = GestureReqognizerOptions(
            base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode=VisionRunningMode.VIDEO,
        )
        self._recognizer = GestureReqognizer.create_from_options(options)

        return {
            self.outputSignal: self._state(),
        }
    
    def step(self, data):
        recording_started = False
        recording_stopped = False
        gesture = "None"
        score = 0.0

        frame = data.get("webcam")

        if frame is None:
            self.current_gesture = "NO WEBCAM"
            self.status_message = "NO FRAME"
            return {
                self.outputSignal: self._state(),
            }

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_rgb.dtype != np.uint8:
            if frame_rgb.max() <= 1.0:
                frame_rgb = (np.clip(frame_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        frame_rgb = np.ascontiguousarray(frame_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        timestamp_ms = int(time.time() * 1000)
        result = self._recognizer.recognize_for_video(mp_image, timestamp_ms)

        if result.gestures:
            category = result.gestures[0][0]
            gesture = category.category_name
            score = category.score

            self.current_gesture = f"{gesture} {score:.2f}"

            # Start recording only when Pointing Up gesture
            if gesture == "Pointing_Up" and not self.is_recording:
                self.is_recording = True
                recording_started = True

            # Stop and save only if recording was already active + Closed Fist gesture
            elif gesture == "Closed_Fist" and self.is_recording:
                self.is_recording = False
                recording_stopped = True
        return {self.outputSignal: {
                "gesture": gesture,
                "score": score,
                "is_recording": self.is_recording,
                "recording_started": recording_started,
                "recording_stopped": recording_stopped
                    }}
    def stop(self, data):
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def _state(self):
        return {
            "gesture": "None",
            "score": 0.0,
            "is_recording": self.is_recording,
            "recording_started": False,
            "recording_stopped": False,
        }