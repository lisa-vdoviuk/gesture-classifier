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
        frame = data["webcam"].key()
        return {}

    def stop(self, data):
        pass