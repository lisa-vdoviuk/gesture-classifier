import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, get_nested_key, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


def draw_hand_landmarks(hand_landmarks, galy: GALY):
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    x = np.inf
    y = np.inf
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (hand_landmarks[conn.start].x,
                     hand_landmarks[conn.start].y)
            end   = (hand_landmarks[conn.end].x,
                     hand_landmarks[conn.end].y)
            x = min(x, start[0], end[0])
            y = min(y, start[1], end[1])
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 5, (255, 255, 255), 1)
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 4, lm[key]["color"], -1)


class HandDetector(Module):
    def __init__(self, outputSignal="detector"):
        super().__init__(
            inputSignals=["config", "webcam"],
            outputSchema={"type":"object", "properties": {outputSignal: {}}},
            name="detector"
        )
        self.outputSignal = outputSignal

    def start(self, data):
        # --- Initialization (called once) ---

        # Load MediaPipe model 
        model_path = get_nested_key("config.detector.modelPath", data)

        # Load handlandmark.task
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"Downloading model to {model_path}...")
            urllib.request.urlretrieve(url, model_path)
            print("Done.")

        # Create BaseOptions
        base_options = python.BaseOptions(model_asset_path = model_path)

        # Initialize HandLandmarkerOptions
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )

        # Create detector and save in self
        self.detector = vision.HandLandmarker.create_from_options(options)

        return {}
    
    def step(self, data):
        #--- One frame ---

        # Get current frame of the camera
        frame = data["webcam"] # np.ndarray, BGR, float32 [0..1]
        H, W = frame.shape[:2]
        
        # Conver BGR -> RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap in mp.Image 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_rgb)

        # Run detector
        result = self.detector.detect(mp_image)
        
        # Create GALY, add layer, draw landmarks
        galy = GALY()
        galy.layer("hands", alwaysVisible=True)

        mapping = np.array([
            [W, 0, 0],
            [0, H, 0],
        ], dtype=np.float64)
        galy.set_layer_affine_mapping(mapping)

        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(hand_landmarks, galy)
        
        # Return result of detection and GALY
        return {self.outputSignal: result, "galy": galy}
    
    def stop(self, data):
        pass

