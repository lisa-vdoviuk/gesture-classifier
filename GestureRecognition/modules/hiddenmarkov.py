import numpy as np
from pathlib import Path

from SignalHub import Module, GALY
from GestureRecognition.hmmclassifier import HMMClassifier

class HiddenMarkov(Module):
    def __init__(self, outputSignal="hmm_prediction"):
        super().__init__(
            inputSignals=["preprocessor", "app_state"],
             outputSchema={"type":"object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )
        self.outputSignal = outputSignal
    
    def start(self, data):
        model_path = Path("data/hmm_classifier.pkl")
        if not model_path.exists():
            self.classifier = None
            return {}
        else:
            self.classifier = HMMClassifier.load(model_path) # load the trained HMM classifier
            return {}
    
    def step(self, data):
        app_state = data["app_state"]
        mode = app_state["mode"]
        if mode != "classification": # only in classification mode
            return {}
        else:
            trajectory = data["preprocessor"]

            galy = GALY()
            galy.layer("hmm-prediction", alwaysVisible=True)

            if self.classifier is None:
                label = "NO HMM"
            else:
                if trajectory is None:
                    label = "-"
                else:
                    label = self.classifier.predict([trajectory])[0] #get the prediciton form hmm model
        
            galy.putText( #draw the letter
                text=label,
                org=(10, 55),
                fontScale=1.5,
                color=(180, 80, 255),
                thickness=3,
                )

            return {self.outputSignal: {"label": label}, "galy": galy}
    
    def stop(self, data):
        return {}