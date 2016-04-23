import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed

def creat_mne_raw_object(fname,read_events):
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
        
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw

subjects = range(1,13)
ids_tot = []
pred_tot = []

# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

# training subsample
# subsample = 10

# output
training_output_dir = 'input/training_preprocessed/'
test_output_dir = 'input/test_preprocessed/'

for subject in subjects:
    print "Loading subject " + str(subject)
    fnames =  glob('input/train/subj%d_series*_data.csv' % (subject))
    fnames.sort()
    raws = map(creat_mne_raw_object, fnames, [True]*len(fnames))
    ids = []
    epochs_tot = []
    y = [] 
    allraws = []   

    picks = pick_types(raws[0].info,eeg=True)
    for i in range(0,len(fnames)):
        raws[i]._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raws[i]._data[j]) for j in picks))
        ids.append(np.array(pd.read_csv(fnames[i])['id']))
        allraws.append(raws[i].copy())
    
    allraws = concatenate_raws(allraws)
    
    ################ CSP Filters training #####################################
    print  "\tTraining CSP"
    # get event posision corresponding to HandStart
    events = find_events(allraws,stim_channel='HandStart', verbose=False)
    # epochs signal for 2 second after the event
    epochs = Epochs(allraws, events, {'during' : 1}, 0, 2, proj=False, baseline=None, preload=True,
                    picks=picks, add_eeg_ref=False, verbose=False)
    
    epochs_tot.append(epochs)
    y.extend([1]*len(epochs))
    
    # epochs signal for 2 second before the event, this correspond to the 
    # rest period.
    epochs_rest = Epochs(allraws, events, {'before' : 1}, -2, 0, proj=False, baseline=None, preload=True,
                    picks=picks, add_eeg_ref=False, verbose=False)
    
    # Workaround to be able to concatenate epochs with MNE
    epochs_rest.times = epochs.times
    
    y.extend([-1]*len(epochs_rest))
    epochs_tot.append(epochs_rest)                                                                                                                                                                                                                                                                                                                                                                                                              
        
    # Concatenate all epochs
    epochs = concatenate_epochs(epochs_tot)
    
    # get data 
    X = epochs.get_data()
    y = np.array(y)                                                                                                                                     
    
    # train csp
    csp = CSP(n_components=nfilters, reg='oas')
    csp.fit(X,y)

    # pre process training and test set. comment or uncomment to only preprocess one
    
    ################ process training set #################################
    # print "\tWriting preprocessed training files"
    # for i in range(0,len(fnames)):
    #     # apply csp filters and rectify signal
    #     feat = np.dot(csp.filters_[0:nfilters],raws[i]._data[picks])**2
    
    #     # smoothing by convolution with a rectangle window    
    #     feattr = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[j],boxcar(nwin),'full') for j in range(nfilters)))
    #     feattr = np.log(feattr[:,0:feat.shape[1]])
        
    #     #write to file
    #     df = pd.DataFrame(data=feattr.T, index=ids[i], columns=[1,2,3,4])
    #     df.to_csv(training_output_dir + 'subj' + str(subject) + '_series' + str(i+1) + '_data' + '_CSP.csv', index_label='id')
    ############################################################################

    ################# process test set ########################################
    print "\twriting preprocessed testing files files"
    ids = []
    fnames =  glob('input/test/subj%d_series*_data.csv' % (subject))
    raws = map(creat_mne_raw_object, fnames, [False]*len(fnames))

    for i in range(0,len(fnames)):
        raws[i]._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raws[i]._data[j]) for j in picks))
        ids.append(np.array(pd.read_csv(fnames[i])['id']))

    for i in range(0,len(fnames)):
        # apply csp filters and rectify signal
        feat = np.dot(csp.filters_[0:nfilters],raws[i]._data[picks])**2
    
        # smoothing by convolution with a rectangle window    
        feattr = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[j],boxcar(nwin),'full') for j in range(nfilters)))
        feattr = np.log(feattr[:,0:feat.shape[1]])
        
        #write to file
        df = pd.DataFrame(data=feattr.T, index=ids[i], columns=[1,2,3,4])
        df.to_csv(test_output_dir + 'subj' + str(subject) + '_series' + str(i+1) + '_data' + '_CSP.csv', index_label='id')


    ############################################################################