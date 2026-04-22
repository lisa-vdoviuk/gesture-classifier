from GestureRecognition.modules.handdetector import HandDetector
from GestureRecognition.modules.preprocessor import Preprocessor
from SignalHub import Engine, ConfigParser, Webcam

initial_data = {"config":{}}

modules = [
    ConfigParser(),
    Webcam(),
    HandDetector(),
    Preprocessor(),
]

engine = Engine(modules=modules, signals=[])
engine.run(initial_data)
