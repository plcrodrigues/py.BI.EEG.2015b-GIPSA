
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from tqdm import tqdm

import sys
sys.path.append('.')
from braininvaders2015b.dataset import BrainInvaders2015b

from scipy.io import loadmat
import numpy as np
import mne

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
 
dataset = BrainInvaders2015b()

scores = {}
for pair in dataset.pair_list:
    scores[pair] = {}

    print('pair', str(pair))

    sessions = dataset._get_single_pair_data(pair=pair)
    for session_name in sessions.keys():

        scores[pair][session_name] = {}

        raw = sessions[session_name]['run_1']

        for subject in [1, 2]:

            if subject == 1:
                pick_channels = raw.ch_names[0:32] + [raw.ch_names[-1]]
            elif subject == 2:
                pick_channels = raw.ch_names[32:-1] + [raw.ch_names[-1]]    

            raw_subject = raw.copy().pick_channels(pick_channels)

            # filter data and resample
            fmin = 1
            fmax = 20
            raw_subject.filter(fmin, fmax, verbose=False)            

            # detect the events and cut the signal into epochs
            events = mne.find_events(raw=raw_subject, shortest_event=1, verbose=False)
            event_id = {'NonTarget': 1, 'Target': 2}
            epochs = mne.Epochs(raw_subject, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
            epochs.pick_types(eeg=True)

            # get trials and labels
            X = epochs.get_data()
            y = epochs.events[:,-1]
            y = y - 1

            # cross validation
            skf = StratifiedKFold(n_splits=5)
            clf = make_pipeline(ERPCovariances(estimator='lwf', classes=[1]), MDM())
            scr = cross_val_score(clf, X, y, cv=skf, scoring = 'roc_auc').mean()
            scores[pair][session_name][subject] = scr

            print('session ' + session_name + ', subject ' + str(subject) + ' : ' + str(scr))

    print('')

filename = 'classification_scores.pkl'
joblib.dump(scores, filename)    
