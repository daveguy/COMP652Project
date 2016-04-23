# calculates scores for a given prediction file, against the given truth file. third parameter is the output file name

import pandas as pd
import sys
import numpy as np
from sklearn.metrics import roc_auc_score as ras

results_file = sys.argv[1]
truth_file = sys.argv[2]
out = sys.argv[3]

results = pd.read_csv(results_file)
truth = pd.read_csv(truth_file)

cols = np.array(truth.columns[1:])

scores = np.empty(6)

for i in range(0,6):
	scores[i] = ras(np.array(truth[cols[i]]), np.array(results[cols[i]]))

avg_score = ras(truth[cols], results[cols], average='macro')

f=open(out, 'w')
f.write('Average AUC score: ' + str(avg_score) + '\n')
f.write('Scores by Event:\n')
f.write('{0:>20} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}'.format(cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]))
f.write('\n')
f.write('{0:20.5} {1:20.5} {2:20.5} {3:20.5} {4:20.5} {5:20.5}'.format(scores[0], scores[1], scores[2], scores[3], scores[4], scores[5]))
