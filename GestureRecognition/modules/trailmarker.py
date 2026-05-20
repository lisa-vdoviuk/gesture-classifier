from SignalHub import Module, get_nested_key, GALY
from collections import deque
import numpy as np
import mediapipe as mp
from pathlib import Path
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
GestureReqognizer = mp.tasks.vision.GestureRecognizer
GestureReqognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class TrailMarker(Module):
    def __init__(self, outputSignal="trailmarker"):
        super().__init__(
            inputSignals=["config","training_controller","detector", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        gesture_model_path = Path("gesture_recognizer.task")
        self.trail = deque(maxlen=60) # Creating a list with max of 60 variables.
        self.lost_frames = 0
        self.max_lost_frames = 5
        self.index_finger = 8
        options = GestureReqognizerOptions(
            base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode=VisionRunningMode.VIDEO,
        )
        self._recognizer = GestureReqognizer.create_from_options(options)

        return {

        }

    def step(self, data):
        result = data["detector"]
        frame = data["webcam"]
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
        gesture_detected = False
        if result.gestures:
            category = result.gestures[0][0]
            gesture = category.category_name
            score = category.score

            self.current_gesture = f"{gesture} {score:.2f}"
            if gesture == "Pointing_Up":
                # Grabs the very first hand it sees (Hand 0)
                if getattr(result, "hand_landmarks", None):
                    hand = result.hand_landmarks[0]
                    x = hand[self.index_finger].x
                    y = hand[self.index_finger].y

                    self.trail.append((x,y))
                    self.lost_frames = 0
                    gesture_detected = True
            elif gesture == "Close_Fist":
                self.trail.clear()
                self.lost_frames = 0
                gesture_detected = True
        if not gesture_detected:
            self.lost_frames+=1
            if self.lost_frames >= self.max_lost_frames:
                self.trail.clear()
        
        galy = GALY()
        galy.layer("trail", alwaysVisible=True)

        # Scales the drawing to match the webcam size (took it from handdetector.py)
        if frame is not None:
            H, W = frame.shape[:2]
            mapping = np.array([
                [W, 0, 0],
                [0, H, 0],
            ], dtype=np.float64)
            galy.set_layer_affine_mapping(mapping)

        points = list(self.trail)
        for i in range(1, len(points)):
            pt1 = points[i-1]
            pt2 = points[i]
            
            galy.line(pt1, pt2, color=(255, 255, 0), thickness=4)

        return {self.outputSignal: {}, "galy": galy}

    def stop(self, data):
        pass