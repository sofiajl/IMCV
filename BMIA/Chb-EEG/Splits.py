from Patient import Patient
import numpy as np

class Splits():

    def patient_splits(patient_id, patient_path, train_frac = 0.8):
        """
        Generate splits for train/test for one given patient
        """

        # Getting patient info
        p = Patient(patient_id, patient_path)
        # Getting subsets
        clips = p.get_seizure_subsets()

        test_frac = 1.0 - train_frac
        n_test_seizures = max(1, int(round(test_frac * len(clips))))
        n_train_seizures = len(clips) - n_test_seizures

        # Building train/test sets
        train_data = np.concatenate([data[0] for data in clips[:n_train_seizures]])
        train_label = np.concatenate([data[1] for data in clips[:n_train_seizures]])

        test_data = np.concatenate([data[0] for data in clips[n_train_seizures:]])
        test_label = np.concatenate([data[1] for data in clips[n_train_seizures:]])

        return (
            # Training
            (train_data, train_label),

            # Test
            (test_data, test_label)
        )