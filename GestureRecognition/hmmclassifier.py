import pickle

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMClassifier:
    # This function creates the classifier and saves the HMM settings.
    def __init__(
            self, 
            n_components=5, 
            covariance_type="diag", 
            n_iter=200, 
            resample_len=40, # Added resampling since different letter can have different amount of points, 40 is just an average
            feature_mode="xy_dxy", # This is for mode switching for experimenting which features to use
            min_covar_options=(1e-2, 5e-2), # Options for covarience regularization, to avoid singular covariance matrix during training
            validation_size=0.25, 
            grid_random_state=42,
        ):
        
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.resample_len = resample_len
        self.feature_mode = feature_mode
        self.min_covar_options = min_covar_options
        self.validation_size = validation_size
        self.grid_random_state = grid_random_state
        self.models = {}
        self.classes_ = []

    # This function trains one HMM model for every gesture class.
    def fit(self, dataset):
        self.models = {}
        self.classes_ = []
        self.best_params_ = {}

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

            best_params = self._select_best_params(clean_sequences) # Runs grid search to find the best parameters for the HMM model

            X, lengths = self._prepare_training_data(clean_sequences) # Runs the resampling and feature extraction on the sequences to prepare them for training

            best_model = GaussianHMM(
                n_components=best_params["n_components"],
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
                min_covar=best_params["min_covar"],
                tol=1e-3,
            )

            try:
                best_model.fit(X, lengths)

                self.models[label] = best_model
                self.classes_.append(label)
                self.best_params_[label] = best_params

                print(
                    f"{label}: n={best_params['n_components']}, "
                    f"min_covar={best_params['min_covar']}"
                )

            except Exception as e:
                print(f"Error occurred while fitting final model for {label}: {e}")
                continue

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
            sequence = self._transform_sequence(sequence) # Transforming before scoring
            scores = []

            for label in self.classes_:
                model = self.models[label]

                try:
                    score = model.score(sequence)

                        # Normalize by sequence length.
                        # This makes scores more comparable for gestures with different number of points.
                    score = score / len(sequence)
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

    # This function predicts the class and also returns a confidence value.
    def predict_with_confidence(self, sequences, temperature=10.0):
        scores = self.decision_function(sequences)

        predictions = []
        confidences = []

        for score_row in scores:
            if np.all(np.isneginf(score_row)):
                predictions.append("None")
                confidences.append(0.0)
                continue

            best_index = int(np.argmax(score_row))
            prediction = self.classes_[best_index]

            # HMM scores are log-likelihoods, so we convert them into a confidence-like value with a softmax.
            stable_scores = score_row - np.max(score_row)
            exp_scores = np.exp(stable_scores / temperature)
            probabilities = exp_scores / np.sum(exp_scores)

            confidence = float(probabilities[best_index])

            predictions.append(prediction)
            confidences.append(confidence)

        return predictions, confidences
    
    # This function saves the trained classifier into a file.
    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    # This function loads a trained classifier from a file.
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    # Resampling function: resamples the sequences to chosen resample_len
    def _resample_sequence(self, sequence):
        sequence = np.asarray(sequence, dtype=float)

        if self.resample_len is None:
            return sequence
        
        if len(sequence) == self.resample_len:
            return sequence
        
        old_t = np.linspace(0.0, 1.0, len(sequence))
        new_t = np.linspace(0.0, 1.0, self.resample_len)

        x_new = np.interp(new_t, old_t, sequence[:,0])
        y_new = np.interp(new_t, old_t, sequence[:,1])

        resampled = np.column_stack([x_new, y_new])

        return resampled

    # Function for extracting features from the sequence based on the selected feature mode
    # For now it simply calculates the differences between consecutive points and concatenates them with the original points
    def _extract_features(self, sequence):
        sequence = np.asarray(sequence, dtype=float)

        if self.feature_mode =="xy":
            return sequence
        if self.feature_mode == "xy_dxy":
            dxy = np.diff(sequence, axis=0, prepend=sequence[:1])
            features = np.concatenate([sequence, dxy], axis=1)
            return features
    
    # Transformer function to apply resampling and feature extraction instead of calling them separately
    def _transform_sequence(self, sequence):
        sequence = self._resample_sequence(sequence)
        sequence = self._extract_features(sequence)
        return sequence

    # This function combines all sequences and also stores their lengths.
    def _prepare_training_data(self, sequences):
        transformed_sequences = []

        for sequence in sequences:
            sequence = self._transform_sequence(sequence)
            transformed_sequences.append(sequence)

        lengths = []

        for sequence in transformed_sequences:
            lengths.append(len(sequence))

        X = np.concatenate(transformed_sequences, axis=0)

        return X, lengths


    # This function selects the best parameters for the HMM model using grid search and cross-validation.
    def _select_best_params(self, sequences):
        sequences = list(sequences)

        # If there are too few samples, do not split.
        # Just use all sequences both for training and validation.
        if len(sequences) < 4:
            train_sequences = sequences
            val_sequences = sequences
        else:
            rng = np.random.default_rng(self.grid_random_state)
            indices = rng.permutation(len(sequences))

            val_count = max(1, int(len(sequences) * self.validation_size))

            val_indices = indices[:val_count]
            train_indices = indices[val_count:]

            train_sequences = [sequences[i] for i in train_indices]
            val_sequences = [sequences[i] for i in val_indices]

        best_score = float("-inf")
        best_params = {
            "n_components": 2,
            "min_covar": self.min_covar_options[0],
        }

        for n in range(2, self.n_components + 1):
            for min_covar in self.min_covar_options:
                try:
                    X_train, train_lengths = self._prepare_training_data(train_sequences)

                    model = GaussianHMM(
                        n_components=n,
                        covariance_type=self.covariance_type,
                        n_iter=self.n_iter,
                        random_state=42,
                        min_covar=min_covar,
                        tol=1e-3,
                    )

                    model.fit(X_train, train_lengths)

                    validation_scores = []

                    for sequence in val_sequences:
                        sequence = self._transform_sequence(sequence)

                        score = model.score(sequence)
                        score = score / len(sequence)

                        validation_scores.append(score)

                    mean_validation_score = np.mean(validation_scores)

                    if mean_validation_score > best_score:
                        best_score = mean_validation_score
                        best_params = {
                            "n_components": n,
                            "min_covar": min_covar,
                        }

                except Exception as e:
                    print(f"Grid search error: n={n}, min_covar={min_covar}, error={e}")
                    continue

        return best_params

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
        y_true = []
        y_pred = []

        for label, sequences in test_dataset.items():
            predictions = self.predict(sequences)
            y_true.extend([label] * len(predictions))
            y_pred.extend(predictions)

        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        print("Classification report:\n")
        print(classification_report(y_true, y_pred, labels=self.classes_, zero_division=0))

        print(f"Overall accuracy: {accuracy_score(y_true, y_pred):.3f}")