import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_PATH = Path("data") / "dataset.pkl"


def load_dataset(dataset_path=DATASET_PATH):
    """Loads the saved dataset from data/dataset.pkl."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Please record data first with: python run_labeling.py"
        )

    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    return dataset


def visualize_dataset(max_samples_per_class=30):
    """
    Shows several recorded gesture trajectories for each class.

    This helps us check if the gestures look different from each other
    and if there are wrong or strange recordings.
    """
    dataset = load_dataset()

    for label, recordings in dataset.items():
        plt.figure(figsize=(6, 6))
        plt.title(f"Gesture class: {label}")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.grid(True)

        shown = 0

        for recording in recordings:
            points = np.array(recording)

            # We expect points like: [[x1, y1], [x2, y2], ...]
            if points.ndim != 2 or points.shape[1] != 2:
                continue

            x_values = points[:, 0]
            y_values = points[:, 1]

            plt.plot(x_values, y_values, marker="o", alpha=0.7)
            shown += 1

            if shown >= max_samples_per_class:
                break

        plt.axis("equal")
        plt.show()


def replay_recordings(label=None, sample_index=0, pause_time=0.2):
    """
    Replays one recorded gesture step by step.
    This is useful to see how the gesture movement was recorded.
    """
    dataset = load_dataset()

    if not dataset:
        print("Dataset is empty.")
        return

    if label is None:
        label = list(dataset.keys())[0]

    if label not in dataset:
        print(f"Label '{label}' not found.")
        print(f"Available labels: {list(dataset.keys())}")
        return

    recordings = dataset[label]

    if sample_index >= len(recordings):
        print(f"Sample index {sample_index} does not exist for label '{label}'.")
        return

    points = np.array(recordings[sample_index])

    if points.ndim != 2 or points.shape[1] != 2:
        print("This recording has the wrong format.")
        return

    plt.figure(figsize=(6, 6))

    for i in range(1, len(points) + 1):
        plt.clf()
        plt.title(f"Replay: {label} | Sample {sample_index}")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.grid(True)
        plt.axis("equal")

        current_points = points[:i]
        plt.plot(current_points[:, 0], current_points[:, 1], marker="o")

        plt.pause(pause_time)

    plt.show()


def evaluate_classifier(y_true=None, y_pred=None):
    """
    Evaluates classifier results.

    y_true = correct labels
    y_pred = predicted labels
    """
    if y_true is None or y_pred is None:
        print("Evaluation is not possible yet.")
        print("The HMM classifier must first return predictions.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        print("y_true and y_pred must have the same length.")
        return

    accuracy = np.mean(y_true == y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    labels = sorted(set(y_true) | set(y_pred))

    print("\nConfusion Matrix")
    print("----------------")
    print("Labels:", labels)

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: i for i, label in enumerate(labels)}

    for true_label, predicted_label in zip(y_true, y_pred):
        row = label_to_index[true_label]
        col = label_to_index[predicted_label]
        matrix[row, col] += 1

    print(matrix)


def print_dataset_info():
    """Prints simple information about the dataset."""
    dataset = load_dataset()

    print("Dataset information")
    print("-------------------")

    for label, recordings in dataset.items():
        valid = 0
        invalid = 0

        for recording in recordings:
            points = np.array(recording)

            if points.ndim == 2 and points.shape[1] == 2:
                valid += 1
            else:
                invalid += 1

        print(f"Class: {label}")
        print(f"Recordings: {len(recordings)}")
        print(f"Valid recordings: {valid}")
        print(f"Invalid recordings: {invalid}")
        print()


if __name__ == "__main__":
    print_dataset_info()
    visualize_dataset()