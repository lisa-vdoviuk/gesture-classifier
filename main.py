from GestureRecognition.modules.handdetector import HandDetector
from GestureRecognition.modules.preprocessor import Preprocessor
from GestureRecognition.modules.recorder import Recorder
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
        Preprocessor(),
        Recorder()
    ]

    engine = Engine(modules=modules, signals=[])
    engine.run(initial_data)

if __name__ == "__main__":
        parser = argparse.ArgumentParser("GestureRecognition")
        run(parser)

