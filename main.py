from GestureRecognition.modules.handdetector import HandDetector
from SignalHub import Engine, ConfigParser, Webcam

initial_data = {"config":{}}

modules = [
    ConfigParser(),
    Webcam(),
    HandDetector(),
]

engine = Engine(modules=modules, signals=[])
engine.run(initial_data)