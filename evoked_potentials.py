
import mne

from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances, ERPCovariances
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

import sys
sys.path.append('.')
from braininvaders2014b.dataset import BrainInvaders2014b

#filename = 'classification_scores.pkl'
#scores = joblib.load(filename)
dataset = BrainInvaders2014b()

# loop on list of subjects
for pair in dataset.pair_list:

	print('treating pair', str(pair).zfill(2))

	# plot the figure
	fig, ax = plt.subplots(facecolor='white', figsize=(17.00, 12.30), nrows=2, ncols=4)

	# get the raw object
	sessions = dataset._get_single_pair_data(pair=pair)
	raw = sessions[session_name]['run_1']
	chname2idx = {}
	for i, chn in enumerate(raw_colab.ch_names):
		chname2idx[chn] = i

	# filter data and resample
	fmin = 1
	fmax = 20
	raw.filter(fmin, fmax, verbose=False)

	# detect the events and cut the signal into epochs
	event_id = {'NonTarget': 1, 'Target': 2}
	events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
	epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)

	for i, epochs, Cz in zip([0, 1], [epochs_solo1, epochs_solo2], ['Cz_1', 'Cz_2']):

		evkTarget = epochs['Target'].average().data[chname2idx[Cz],:]
		evkNonTarget = epochs['NonTarget'].average().data[chname2idx[Cz],:]
		t = np.arange(len(evkTarget)) / epochs_solo1.info['sfreq']
		ax[0][i].plot(t, evkTarget, c='#2166ac', lw=3.0, label='Target (' + str(len(epochs['Target'])) + ' trials)')
		ax[0][i].plot(t, evkNonTarget, c='#b2182b', lw=3.0, label='NonTarget (' + str(len(epochs['NonTarget'])) + ' trials)')
		ax[0][i].plot([0, 0.8], [0, 0], c='#CDCDCD', lw=2.0, ls='--')	
		ax[0][i].set_xlim(0, 0.8)
		ax[0][i].set_title('subject ' + str(i+1) + ' on solo', fontsize=12)
		ax[0][i].set_ylabel(r'amplitude ($\mu$V)', fontsize=10)
		ax[0][i].set_xlabel('time after stimulus (s)', fontsize=10)
		ax[0][i].legend()		

	for i, Cz in zip([0, 1], ['Cz_1', 'Cz_2']):

		evkTarget = epochs_colab['Target'].average().data[chname2idx[Cz],:]
		evkNonTarget = epochs_colab['NonTarget'].average().data[chname2idx[Cz],:]
		t = np.arange(len(evkTarget)) / epochs_solo1.info['sfreq']
		ax[1][i].plot(t, evkTarget, c='#2166ac', lw=3.0, label='Target (' + str(len(epochs['Target'])) + ' trials)')
		ax[1][i].plot(t, evkNonTarget, c='#b2182b', lw=3.0, label='NonTarget (' + str(len(epochs['NonTarget'])) + ' trials)')
		ax[1][i].plot([0, 0.8], [0, 0], c='#CDCDCD', lw=2.0, ls='--')	
		ax[1][i].set_xlim(0, 0.8)
		ax[1][i].set_title('subject ' + str(i+1) + ' on collaborative', fontsize=12)
		ax[1][i].set_ylabel(r'amplitude ($\mu$V)', fontsize=10)
		ax[1][i].set_xlabel('time after stimulus (s)', fontsize=10)		
		ax[1][i].legend()
	
# fig.suptitle('Average evoked potentials at electrode Cz for pair ' + str(pair), fontsize=14)
# filename = './evoked_potentials/evoked_potentials_pair_' + str(pair).zfill(2) + '.pdf'
# fig.savefig(filename, format='pdf')

