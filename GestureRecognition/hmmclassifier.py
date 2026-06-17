import pickle

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    # This function creates the classifier and saves the HMM settings.
    def __init__(self, n_components=5, covariance_type="diag", n_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self.models = {}
        self.classes_ = []

    # This function trains one HMM model for every gesture class.
    def fit(self, dataset):
        self.models = {}
        self.classes_ = []

        for label in sorted(dataset.keys()):
            sequences = dataset[label]
            clean_sequences = []

            for sequence in sequences:
                sequence = np.asarray(sequence, dtype=float)

                if self._is_valid_sequence(sequence):
                    clean_sequences.append(sequence)

            if len(clean_sequences) < 2:
                print(f"Skipping {label}: not enough valid samples")
                continue

            X, lengths = self._prepare_training_data(clean_sequences)

            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )

            model.fit(X, lengths)

            self.models[label] = model
            self.classes_.append(label)

            print(f"Trained HMM for {label} with {len(clean_sequences)} samples")

        if len(self.models) == 0:
            raise ValueError("No models were trained. Please check the dataset.")

        return self

    # This function calculates the score of each sequence for every trained class.
    def decision_function(self, sequences):
        if len(self.models) == 0:
            raise ValueError("The classifier is not trained yet.")

        sequences = self._make_sequence_list(sequences)
        all_scores = []

        for sequence in sequences:
            sequence = np.asarray(sequence, dtype=float)

            if not self._is_valid_sequence(sequence):
                all_scores.append([float("-inf")] * len(self.classes_))
                continue

            scores = []

            for label in self.classes_:
                model = self.models[label]

                try:
                    score = model.score(sequence)
                except Exception:
                    score = float("-inf")

                scores.append(score)

            all_scores.append(scores)

        return np.array(all_scores)

    # This function predicts the class with the highest HMM score.
    def predict(self, sequences):
        scores = self.decision_function(sequences)
        best_indices = np.argmax(scores, axis=1)

        predictions = []

        for index in best_indices:
            predictions.append(self.classes_[index])

        return predictions

    # This function saves the trained classifier into a file.
    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    # This function loads a trained classifier from a file.
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    # This function combines all sequences and also stores their lengths.
    def _prepare_training_data(self, sequences):
        lengths = []

        for sequence in sequences:
            lengths.append(len(sequence))

        X = np.concatenate(sequences, axis=0)

        return X, lengths

    # This function makes sure that one sequence and many sequences are handled correctly.
    def _make_sequence_list(self, sequences):
        if isinstance(sequences, np.ndarray):
            if sequences.ndim == 2:
                return [sequences]

        return list(sequences)

    # This function checks if a gesture trajectory has the correct format.
    def _is_valid_sequence(self, sequence):
        if sequence is None:
            return False

        if sequence.ndim != 2:
            return False

        if sequence.shape[1] != 2:
            return False

        if len(sequence) < 2:
            return False

        if np.isnan(sequence).any():
            return False

        return True


    # This function splits the dataset into training data and test data.
    def train_test_split_dataset(self, dataset, test_size=0.2, random_state=42):
        rng = np.random.default_rng(random_state)

        self.train_dataset = {}
        self.test_dataset = {}

        for label, sequences in dataset.items():
            sequences = list(sequences)

            if len(sequences) < 2:
                continue

            indices = rng.permutation(len(sequences))
            test_count = max(1, int(len(sequences) * test_size))

            if test_count >= len(sequences):
                test_count = len(sequences) - 1

            test_indices = indices[:test_count]
            train_indices = indices[test_count:]

            self.train_dataset[label] = [sequences[i] for i in train_indices]
            self.test_dataset[label] = [sequences[i] for i in test_indices]

        return self.train_dataset, self.test_dataset


    # This function tests the classifier and prints the accuracy.
    def evaluate_classifier(self, test_dataset):
        self.correct = 0
        self.total = 0

        for label, sequences in test_dataset.items():
            predictions = self.predict(sequences)

            for prediction in predictions:
                if prediction == label:
                    self.correct += 1

                self.total += 1

        if self.total == 0:
            print("No test data available.")
            return 0

        self.accuracy = self.correct / self.total
        print(f"Accuracy: {self.accuracy:.2f}")

        return self.accuracy