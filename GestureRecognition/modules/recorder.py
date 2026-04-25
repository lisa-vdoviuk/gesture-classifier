from SignalHub import GALY, get_nested_key, Module
import numpy as np
import os
class Recorder(Module):
    def __init__(self, outputSignal="recorder"):
        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="recorder",
        )
        self.outputSignal = outputSignal
    def start(self,data):
        config = data["config"]
        self.file_path = config["recorder"]["file"]
        folder_path = os.path.dirname(self.file_path)
        os.makedirs(folder_path, exist_ok=True)
        self.saved = False
        return {}
    def step(self,data):
        trajectory = data["preprocessor"]
        if self.saved == True:
            return {self.outputSignal: self.saved}
        if trajectory is not None:
            np.save(self.file_path, trajectory)
            self.saved = True
            return {self.outputSignal: self.saved}
        return {self.outputSignal: self.saved}