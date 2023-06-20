## Application of Machine Learning To Epileptic Seizure Detection
Ali Shoeb and John Guttag

### **Goal**
Evaluate a machine learning approach, Support Vector Machine, for constructing patient-specific classifiers that detect the onset of an epileptic seizure through analysis of the scalp EEG.

### **Data**
This database, consists of EEG recordings from pediatric subjects with intractable seizures.

Recordings, grouped into 23 cases, were collected from 22 patients (5 males, ages 3–22; and 17 females, ages 1.5–19). Subjects were monitored for up to several days following withdrawal of anti-seizure medication to characterize seizures and assess their candidacy for surgical intervention. In all, the onsets and ends of 182 seizures are annotated.

In total, 916 hours of continuous scalp EEG were sampled at 256 Hz samples per second. Most files contain 23 EEG signals (24 or 26 in a few cases).

### **Approach**
#### Dataset
* The first 20 seconds of each seizure
* At least, 24 hours of non-seizure data

#### Training
* SVMLight software package (Joachims, 1999)
* RBF kernel (non-linear)
* Hyperparameters: gamma = 0.1 and error C = 1.0

#### Testing
* Leave-one-record-out cross-validation scheme


### **Limitations and alternatives**

#### <u>SVM light library</u><br>
*  **Limitation.** It wans't possible to install even following all steps and trying different ways looking into forums. Official library documentation: https://www.cs.cornell.edu/people/tj/svm_light/

* **Alternative.** SVM from sklearn library taken for training instead of svm light -> Memory limitations

#### <u>Input data</u><br>
* **Limitation.** Data for training is massive, even considering a single patient, one of the smallest examples, the dimension is around 23 channels x 15M of points. An SVM is a quadratic problem, also, it needs to store in memory the vectors for the calculations. So, for 50K registers, this algorithm has no capacity for processing in the machine available.

* **Alternative 1.** The original approach is to take the first 20 seconds of each seizure and, at least, 24 hours of non-seizure. In order to reduce this dimension, searching I found a paper that is taking the same data for the study and does a comparison of the classifier. For that, the authors propose a solution which is to take subsets, where giving N minutes, each subset is built by "fetching  N/2  minutes of EEG  data  before the start of a seizure and N/2 minutes of EEG data after the end of a seizure." Thus, in a more visual way, we would have subsets as follow:

            N/2     Si     N/2   ; S represents the seizure and i the time of it

    The author says that "[...] The advantage of this simple training set acquisition is that although the training sets are simpler they are still effective since they contain the most important features which are the original seizure data and some non-seizure data located before and after the seizures[...]". Source: https://www.researchgate.net/publication/307633115_A_novel_method_of_EEG_data_acquisition_feature_extraction_and_feature_space_creation_for_Early_Detection_of_Epileptic_Seizures

* **Alternative 2.** Even after trying the first alternative, the data was still big for training the SVM, so, as a second alternative a subsample of 50k random registers was taking for training and 20k for testing.

#### <u>Leave-One-Out Cross-validation</u><br>
* **Limitation.** As the method itself claims in the documentation:
    "Due to the high number of test sets (which is the same as the number of samples) this cross-validation method can be very costly. For large datasets one should favor KFold, ShuffleSplit or StratifiedKFold." 
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut

* **Alternative.** A StratifiedKFold was used instead of Leave-One-Out cross-validator.


### **Classes and functions**

#### <u>ChbEdfFile class</u><br>

Edf files reader and features extractor.

Functions inside the class:
* get_n_channels: number of channels
* get_n_data_points: number of points of temporal-series
* get_channel_names: names of the channesl
* get_channel_scalings: scalings of each channel, difference between min and max scale
* get_file_duration: duration in seconds of the EEG sample
* get_sampling_rate: sample rate (Hz) of the sample
* get_channel_data: data for a specific channel
* get_data: getting all the data of the EEG
* get_start_datetime: datetime when the sample started to be gotten
* get_end_datetime: datetime when the sample finished to be gotten


#### <u>ChbMetadata class</u><br>

Metadata seizure extractor for EEG data.

Functions inside the class:
* _parse_file: parse the summary seizure and non-seizures file
* _parse_metadata: parse a single metadata block with the seizure info
* _parse_file_metadata: parse the file metadata list blocks to get the seizure intervals
* get_channel_names: eturn the channel names
* get_seizure_list: get list of seizure intervals for each file
* get_file_metadata: get the metadata for all of the files


#### <u>Patient class</u><br>
Data related to a single patient: EEG data and labels regarding to seizure and non-seizure.

Functions inside the class:
* get_eeg_data: Get the entire EEG series
* get_seizures: Get list of seizure for each sample (seconds)
* get_seizure_intervals: Get list of seizure intervals (begin and end) considering the entire series and the frequency (Hz)
* get_labels: Get labels of the series
* get_seizure_clips: Get subsets (clips); the first clip would be from the beginning to the first half of the first seizure, the second clip from the second half of the first seizure to the first half of the second seizure, and so on until the end of the series.
* get_seizure_subsets: Get subsets of the series without losing seizures, in order to have intervals of time as follow:
                       N/2 + seizure + N/2
                       *Note*: Base on a paper explained in "Limitations and alternatives"

#### <u>Splits class</u><br>
Generate splits for training and testing.

Functions inside the class:
* patient_splits: Generate splits for train/test for one given patient


*Note: Original source https://github.com/dougkoch/chb-mit/tree/ecf7b90a630b029d93dd5faf115eb099291c5589. Some updates have being done in the original classes and also new ones were added.*