import numpy as np
from pathlib import Path
import logging
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
        self.classifier = None
    
    def start(self, data):
        self.classifier = None
        self.loaded_model_version = -1
        return {}
    
    def step(self, data):
        app_state = data["app_state"]
        model_path = Path("data/hmm_classifier.pkl")
        mode = app_state["mode"]
        model_version = app_state.get("model_version", 0)
        if model_version != self.loaded_model_version and model_path.exists():
            try:
                self.classifier = HMMClassifier.load(model_path) # load the trained HMM classifier
                self.loaded_model_version = model_version
            except Exception as e:
                print(f"Could not reload HMM model: {e}")
        if mode != "classification": # only in classification mode
            return {}
        else:
            trajectory = data["preprocessor"]

            galy = GALY()
            galy.layer("hmm-prediction", alwaysVisible=True)

            label = "-"
            confidence = 0.0

            if self.classifier is None:
                label = "NO HMM"

            elif trajectory is None:
                label = "-"

            else:
                try:
                    predictions, confidences = self.classifier.predict_with_confidence([trajectory])
                    
                    label = predictions[0]
                    confidence = confidences[0]

                except Exception as e:
                    print(f"HMM prediction error: {e}")
                    label = "ERROR"
                    confidence = 0.0
        
            galy.putText( #draw the letter
                text=label,
                org=(10, 55),
                fontScale=1.5,
                color=(180, 80, 255),
                thickness=3,
                )

            confidence_text = f"Confidence: {confidence * 100:.1f}%"

            galy.putText(
                text=confidence_text,
                org=(10, 85),
                fontScale=0.45,
                color=(180, 80, 255),
                thickness=1,
                )

            return {
                self.outputSignal: {
                    "label": label,
                    "confidence": confidence,
                },
                "galy": galy,
            }
    
    def stop(self, data):
        return {}