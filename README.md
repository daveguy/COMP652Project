# COMP652Project

The dataset is not included due to space limitations on upload size, it can be obtained at https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data a kaggle membership is required

preprocessing.py and preprocessing_all.py both perform CSP. the difference is that preprocessing.py only uses "HandStart" events as positive samples. preprocessing_all.py use all events as samples.

SVM.py performs svn classification

get_score.py calculates AUROC score for a single result

get_mean_score.py processes all outputs and calculates mean scores per subject and mean scores overall