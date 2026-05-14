from SignalHub import Module, get_nested_key, GALY
from collections import deque
import numpy as np

class TrailMarker(Module):
    def __init__(self, outputSignal="trailmarker"):
        super().__init__(
            inputSignals=["config","detector", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )
        self.outputSignal = outputSignal

    def start(self, data):
        self.trail = deque(maxlen=60) # Creating a list with max of 60 variables.
        self.lost_frames = 0
        self.max_lost_frames = 5
        self.index_finger = 8
        return {}

    def step(self, data):
        result = data["detector"]
        frame = data["webcam"]

        if result and getattr(result, "hand_landmarks", None):
            # Grabs the very first hand it sees (Hand 0)
            hand = result.hand_landmarks[0]
            x = hand[self.index_finger].x
            y = hand[self.index_finger].y

            self.trail.append((x,y))
            self.lost_frames = 0
        else:
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