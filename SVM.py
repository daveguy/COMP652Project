from sklearn.svm import SVC
import pandas as pd
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import sys

def get_results(i):
	#find all "positive" samples and an equal number of "negative" samples to build data set
	picks = picks = np.array(training_truth[:,i] == 1) #indices where ith class is 1
	zeros = np.logical_not(picks) # indices where ithclass is 0
	X = training_set[picks,:] #data set of positive values

	labels = [1]*X.shape[0]
	
	rang = np.array(range(0,picks.size))
	select = rang[zeros]
	np.random.shuffle(select)
	select = select[:X.shape[0]] #indices for random selection of negative values
	neg_selection = training_set[select,:] # select a subset of "negative" values
	X = np.concatenate((X,neg_selection))
	labels.extend([0]*neg_selection.shape[0])
	labels = np.array(labels)

	print '\ttraining and testing class ' + str(i)
	model  = SVC(kernel='rbf')
	model.fit(X[::subsample,:], labels[::subsample])
	return model.predict(test_set)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
ifall = ''
if len(sys.argv) == 4:
	ifall = sys.argv[3]

subjects = range(1,13)
series = np.array([1,2,3,4,5,6,7,8])
subsample = 10
in_cols = np.array(['1','2','3','4'])
out_cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

for subject in subjects:
	print 'loading subject: ' + str(subject)
	for num in series:
		training_set = []
		training_truth = []
		test_set = []
		test_ids = []

		##read and concatinate all the training series
		fnames_train = glob('input/' + input_dir + '/subj%d_series'% (subject) + str(series[series != num]) + '_data_CSP' + ifall + '.csv' )
		fnames_train.sort()
		fnames_truth = glob('input/train/subj%d_series' %(subject) + str(series[series != num])  + '_events.csv')
		fnames_truth.sort()
		for i in range(0, len(fnames_train)):
			data = pd.read_csv(fnames_train[i])
			training_set.append(np.array(data[in_cols]))
			data = pd.read_csv(fnames_truth[i])
			training_truth.append(np.array(data[out_cols]))

		training_set = np.concatenate(training_set)
		training_truth = np.concatenate(training_truth)

		#read the testing data
		data = pd.read_csv('input/' + input_dir + '/subj%d_series'% (subject) + str(num) + '_data_CSP' + ifall + '.csv')
		test_set = np.array(data[in_cols])
		test_ids = data['id']

		#Train and test SVR
		results = np.empty((len(test_ids),6))
		print '\tTraining and testing SVR'

		results = np.stack(Parallel(n_jobs=-1)
			(delayed(get_results)(i) for i in range(0,6)), 1)

		#save results
		print '\t saving results'
		df = pd.DataFrame(data=results, index=test_ids, columns=out_cols)
		df.to_csv(output_dir + '/subj' + str(subject) + '_series' + str(num) + '_results.csv', index_label='id')


		# zeros = np.logical_not(np.any(training_truth,1)) # indices where all classes are 0
		# X = []
		# labels = []
		# for i in range(0,6):
		# 	#find all "positive" samples and an equal number of "negative" samples to build data set
		# 	picks = np.array(training_truth[:,i] == 1) #indices where ith class is 1
		# 	X .append(training_set[picks,:]) #data set of positive values
		# 	labels.extend([i+1]*picks[picks==True].size)

		# X = np.concatenate(X)
		
		# rang = np.array(range(0,training_truth.shape[0]))
		# select = rang[zeros]
		# np.random.shuffle(select)
		# select = select[:X.shape[0]] #indices for random selection of negative values
		# neg_selection = training_set[select,:] # select a subset of "negative" values

		# X = np.concatenate((X,neg_selection))
		# labels.extend([0]*neg_selection.shape[0])
		# labels = np.array(labels)

		# print '\ttraining and testing'
		# model  = SVC(kernel='rbf')
		# model.fit(X[::subsample,:], labels[::subsample])
		# results = model.predict(test_set)
		
		# expanded_results = np.zeros((len(test_ids),6))
		# for i in range(1,7):
		# 	expanded_results[results==i, i-1] = 1