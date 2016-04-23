'''
Loads CSV files from feature extractions into a format that can be fed to the CNN
Writes periodically to disk to avoid blowing up ram
'''

import sys, os, time, csv, glob, random
import numpy as np

def BinArrayToDec(line):
    n = ''
    for b in line:
        n += str(b)
    return int(n,2)

class FeatureLoader:

    def __init__(self, num_channels, window_size, num_events, val_split_ratio):
        self.num_channels = num_channels
        self.num_events = num_events
        self.window_size = window_size
        self.val_split = val_split_ratio
        
    '''
    Helper Loads a numpy array file
    '''
    def load_numpy_array(self, filename):
        return np.load(filename)
     
    '''
    get_datafiles:
    Call to glob, returns a list of file names matching regex filenames
    '''
    def get_datafiles(self,filenames):
        return glob.glob(filenames)
    
    '''
    import_serie:
    Reads datapoints from file
    '''
    def import_series(self,filename):
        lines = []
        with open(filename) as file:
            raw = csv.reader(file)
            c = 0
            for l in raw: 
                if (c != 0):#Discard first line
                    # Check if a line has the right number of entries
                    if len(l) == self.num_channels + 1:
                        lines.append([float(a) for a in l[1:]])
                c += 1
        return lines
        
    '''
    import_serie:
    Reads datapoints from file
    '''
    def import_events(self,filename):
        lines = []
        with open(filename) as file:
            raw = csv.reader(file)
            c = 0
            for l in raw: 
                if (c != 0):#Discard first line
                    # Check if a line has the right number of entries
                    if len(l) == self.num_events + 1:
                        lines.append([int(a) for a in l[1:]])
                c += 1
        return lines
        
    '''
    to_numpy_tensor:
    Returns a 4D tensor to be used with Theano.
    1st dim are datapoints, 2nd dim is just for 1 channel,
    3rd and 4th dims are the datapoints 2D matrices     
    '''
    def to_numpy_tensor(self, examples):
        l = len(examples)
        X = np.empty([l ,1, self.window_size, self.num_channels],dtype=np.float32)
        for i in range(l):
            X[i][0] = examples[i]
        return X
           
    def load_serie(self, train_path, serie_name, write=True):
        series = []
        
        data_files = [f for f in self.get_datafiles(train_path) if ("data" in f)]
        event_files = [f for f in self.get_datafiles(train_path) if ("events" in f)]
        data_files = [f for f in data_files if (serie_name in f)]
        event_files = [f for f in event_files if (serie_name in f)]
        
        datapoints = []
        for file in data_files:
            lines = self.import_series(file)
            for i in range(0,len(lines),self.window_size):
                if i + self.window_size < len(lines):
                    datapoints.append(np.array(lines[i:i+self.window_size]))
            
        series = self.to_numpy_tensor(datapoints)
        
        events = []
        for file in event_files:
            lines = self.import_events(file)
            for i in range(0,len(lines),self.window_size):
                if i + self.window_size < len(lines):
                    events.append(BinArrayToDec(lines[i]))
        
        zipped = zip(series,events)
        
        random.shuffle(zipped)
        
        series, values = zip(*zipped)
        
        X_train = self.to_numpy_tensor(series[:-int(0.2*len(series))])
        X_test = self.to_numpy_tensor(series[-int(0.2*len(series))+1:])

        Y_train = np.array(values[:-int(0.2*len(values))])
        Y_test = np.array(values[-int(0.2*len(values))+1:])

        if (write==True):
            np.save('X_train_'+serie_name+'.npy',X_train)
            np.save('Y_train_'+serie_name+'.npy',Y_train)
            np.save('X_test_'+serie_name+'.npy',X_test)
            np.save('Y_test_'+serie_name+'.npy',Y_test)

        return X_train,Y_train,X_test,Y_test

F = FeatureLoader(32,150,6,0.2)
F.load_serie("../input/train/*","subj1",True)