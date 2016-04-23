import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as ras

subjects = range(1,13)
series = range(1,9)
#["hand"start","secondThing"....."Average of all"]
cols = ['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']
totalScores = np.zeros(7)
for subject in subjects:
	scores = np.zeros(7)
	print 'calculating scores for subject: ' + str(subject)
	for serie in series:
		data = pd.read_csv('SVM_results_binary_allCSP/subj%d_series%d_results.csv'%(subject, serie))
		data = np.array(data[data.columns[1:]])
		truth = pd.read_csv('input/train/subj%d_series%d_events.csv'%(subject, serie))
		truth = np.array(truth[cols])

		
		for i in range(0,6):
			scores[i] += ras(truth[:,i], data[:,i])

		scores[6] += ras(truth, data, average='macro')

	scores = np.true_divide(scores, len(series))
	totalScores += scores

	print 'Writing scores for subject: ' + str(subject)
	f = open('SVM_scores/subj%d_mean_scores.txt'%(subject), 'w')
	f.write('Average AUC score: {}\n'.format(scores[6]))
	f.write('Scores by Event:\n')
	f.write('{0:>20} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]))
	f.write('\n')
	f.write('{0:20.5} {1:20.5} {2:20.5} {3:20.5} {4:20.5} {5:20.5}'.format(scores[0], scores[1], scores[2], scores[3], scores[4], scores[5]))

print 'Calculating overall mean scores'
totalScores = np.true_divide(totalScores, len(subjects))
print 'Writing mean scores for all subjects'
f = open('SVM_scores/All_mean_scores.txt', 'w')
f.write('Average AUC score: {}\n'.format(totalScores[6]))
f.write('Scores by Event:\n')
f.write('{0:>20} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]))
f.write('\n')
f.write('{0:20.5} {1:20.5} {2:20.5} {3:20.5} {4:20.5} {5:20.5}'.format(totalScores[0], totalScores[1], totalScores[2], totalScores[3], totalScores[4], totalScores[5]))