import numpy as np
from hmmlearn import hmm
import joblib
import os

class HMMClassifier:
    def __init__(self, n_components=7, covariance_type='diag', n_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}
        self.classes = []

    
    def fit(self, X_dict):
        """ X_dict: dict like {'A': [seq1, seq2,...], 'B': [seq1, seq2,..]}
            where every seq is np.array of shape (T, features)
        """
        self.classes = list(X_dict.keys())
        for label in self.classes:
            sequences = X_dict[label]
            X_train = np.vstack(sequences)
            lengths = [len(s) for s in sequences]

            model = hmm.GaussianHMM(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    n_iter=self.n_iter,
                                    random_state=42)
            print(f"Training HMM for class {label} with {len(sequences)} sequences...")
            model.fit(X_train, lengths)
            self.models[label] = model
        
        print("Evaluating training performance...")
        for label in self.classes:
            sequences = X_dict[label]

            predictions = self.predict(sequences)

            correct_predictions = sum(p == label for p in predictions)
            total_sequences = len(sequences)

            if total_sequences > 0:
                accuracy = correct_predictions / total_sequences
                print(f"Class {label}: {correct_predictions}/{total_sequences} correct, Accuracy: {accuracy:.2f}")
            else:
                print(f"Class {label}: No sequences to evaluate.")
        
        return self
    
    def decision_function(self, X_list):
        """Here we are counting logarithm of likelihood (Log-Likelihood) for each sequence.
           This is made to select which model has to be used for prediction. The higher the Log-Likelihood, the better the model fits the data.
           X_list: list of sequences, where every seq is np.array of shape (T, features)
        """
        n_samples = len(X_list)
        n_classes = len(self.classes)

        # Matrix for results
        scores = np.zeros((n_samples, n_classes))

        for i, seq in enumerate(X_list):
            for j, label in enumerate(self.classes):
                model = self.models[label]
                try:
                    # score() return Log-Likelihood of the sequence under the model log P(O|λ)
                    scores[i,j]=model.score(seq)
                except:
                    # if sequence is too short or "broken"
                    scores[i,j]=-np.inf 
        return scores
    
    def predict(self, X_list):
        """Chooses the class with the highest score"""

        # Get scores matrix
        scores = self.decision_function(X_list)

        # Get index of the best class for each sequence
        best_indices = np.argmax(scores, axis=1)

        # Return the corresponding class labels
        return [self.classes[idx] for idx in best_indices]
    
    def save(self, filename):
        os.makedirs("models", exist_ok=True)
        path = os.path.join("models", os.path.basename(filename))
        joblib.dump(self, path)
        print(f"HMMClassifier saved to {path}")
    
    @staticmethod
    def load(filename):
        return joblib.load(filename)
            