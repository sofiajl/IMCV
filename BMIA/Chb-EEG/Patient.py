import glob
import numpy as np
from ChbEdfFile import ChbEdfFile
from ChbMetadata import ChbMetadata

class Patient:
    """
    Get all the info related to a patient
    """
    def __init__(self, id, path_files):
        self._id = id
        self._path_files = path_files
        files = glob.glob(self._path_files + "*.edf")
        self.sampling_rate = 256
        
        # Reading edf files
        self._edf_files = list(map(
            lambda filename: ChbEdfFile(filename, self._id),
            files
        ))
        
        # Acumulating the duration of each sample
        self._cumulative_duration = [0]
        for file in self._edf_files[:-1]:
            self._cumulative_duration.append(self._cumulative_duration[-1] + file.get_file_duration())

        # Total duration of the samples
        self._duration = (self._cumulative_duration[-1] + self._edf_files[-1].get_file_duration()) * self.sampling_rate
        
        # Seizure list from summary file
        self._seizure_list = ChbMetadata(self._path_files + "chb%02d-summary.txt" % (self._id)).get_seizure_list()[:-2]
        
        # Seizure intervals (begin and end) considering the entire series and the frequency (Hz)
        self._seizure_intervals = []
        for i, file in enumerate(self._seizure_list):
            for seizure in file:
                begin = (seizure[0] + self._cumulative_duration[i]) * self.sampling_rate
                end = (seizure[1] + self._cumulative_duration[i]) * self.sampling_rate
                self._seizure_intervals.append((begin, end))


    def get_eeg_data(self):
        """
        Get the entire EEG series
        """
        for i, file in enumerate(self._edf_files):
            print ("Reading EEG data from file %s" % file._filename)
            if not i:
                data = file.get_data()
            else:
                data = np.vstack((data, file.get_data()))

        return data

    def get_seizures(self):
        """
        Get list of seizure for each sample (seconds)
        """
        return self._seizure_list

    def get_seizure_intervals(self):
        """
        Get list of seizure intervals (begin and end) considering the entire series and the frequency (Hz)
        """
        return self._seizure_intervals

    def get_labels(self):
        """
        Get labels of the series
        """
        labels = np.zeros(self._duration)

        for i, interval in enumerate(self._seizure_intervals):
                labels[int(interval[0]):int(interval[1])] = 1

        return labels

    def get_seizure_clips(self):
        """
        """
        clips = []
        data = self.get_eeg_data()
        labels = self.get_labels()

        for i in range(len(self._seizure_intervals)):
            
            if not i:
                left = 0
            else:
                left = (self._seizure_intervals[i-1][1] + self._seizure_intervals[i][0]) // 2
            if i == len(self._seizure_intervals) - 1:
                right = -1
            else:
                right = (self._seizure_intervals[i][1] + self._seizure_intervals[i+1][0]) // 2
            clips.append((data[int(left):int(right)], labels[int(left):int(right)]))
        
        return clips
    
    def get_seizure_subsets(self, N = 5):
        """
        Get subsets of the series without losing seizures, in order to have intervals of time as follow:
            N/2 + seizure + N/2
        
        Input:
            - N, int, minutes to be taking
        Output:
            - clips, list, data and labels for the interval specified
        """
        
        # Getting the data
        data = self.get_eeg_data()
        # Getting the labels
        labels = self.get_labels()

        # Clips of periods (N/2 + seizure + N/2)
        clips = []
        for seizure in self._seizure_intervals:
            
            left = int(seizure[0] - N * 60 * self.sampling_rate/2)
            right = int(seizure[1] + N * 60 * self.sampling_rate/2)
            
            clips.append((data[left: right], labels[left: right]))
        
        return clips