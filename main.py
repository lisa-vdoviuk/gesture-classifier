from GestureRecognition.modules.gesturestate import GestureState
from GestureRecognition.modules.handdetector import HandDetector
from GestureRecognition.modules.preprocessor import Preprocessor
from GestureRecognition.modules.trainingcontroller import TrainingController
from GestureRecognition.modules.trailmarker import TrailMarker
from GestureRecognition.modules.hiddenmarkov import HiddenMarkov
from SignalHub import Engine, ConfigParser, Webcam
import argparse


initial_data = {"config":{}}

def run(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", action="store", default="none")
    parser.add_argument("--recorder.file", action="store")
    modules = [
        ConfigParser(parser),
        Webcam(),
        HandDetector(),
        GestureState(),
        Preprocessor(),
        TrainingController(),
        TrailMarker(),
        HiddenMarkov(),

        
    ]

    engine = Engine(modules=modules, signals=[])
    engine.run(initial_data)

if __name__ == "__main__":
        parser = argparse.ArgumentParser("GestureRecognition")
        run(parser)
