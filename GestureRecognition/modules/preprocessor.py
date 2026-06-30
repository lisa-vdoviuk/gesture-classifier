from SignalHub import GALY, get_nested_key, Module
from collections import deque
import numpy as np

class Preprocessor(Module):
    def __init__(self, outputSignal="preprocessor"):
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="preprocessor",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        self.trajectory = deque(maxlen=70)
        self.lost_frames = 0
        self.min_points = 20
        self.max_lost_frames = 4
        self.wrist_idx = 0
        self.index_idx = 8
        return {}
    def step(self, data):
        result = data["detector"]
        if result and getattr(result, "hand_landmarks", None):
            hand = result.hand_landmarks[0]
            x_rel = hand[8].x 
            y_rel = -hand[8].y
            self.trajectory.append([x_rel,y_rel]) # tracks index fingertip
            self.lost_frames = 0
            if len(self.trajectory) < self.min_points:
                return {self.outputSignal: None}
            else:
                arr = np.array(list(self.trajectory))
                center = arr.mean(axis=0)
                arr_centered = arr - center
                max_arr = np.max(np.abs(arr_centered))
                if max_arr > 0:
                    scaled_arr = arr_centered/max_arr
                    return {self.outputSignal: scaled_arr} # center normalization + scaling
                else:
                    return {self.outputSignal: None}
                
        else:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                deque.clear(self.trajectory) # basic lost-frame handling (hand not in the frame)
            return {self.outputSignal: None}
    def stop(self, data):
        pass