#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import mne
import numpy as np
from braininvaders2015b import download as dl
import os
import glob
import zipfile
import yaml
from scipy.io import loadmat
from distutils.dir_util import copy_tree
import shutil
import pandas as pd

BI2015b_URL = 'https://zenodo.org/record/3268762/files/'

class BrainInvaders2015b():
    '''
    This dataset contains electroencephalographic (EEG) recordings 
    of 44 subjects playing in pair to the multi-user version of a visual 
    P300 Brain-Computer Interface (BCI) named Brain Invaders. The interface 
    uses the oddball paradigm on a grid of 36 symbols (1 or 2 Target, 
    35 or 34 Non-Target) that are flashed pseudo-randomly to elicit the 
    P300 response. EEG data were recorded using 32 active wet electrodes 
    per subjects (total: 64 electrodes) during four randomised conditions 
    (Cooperation 1-Target, Cooperation 2-Targets, Competition 1-Target, 
    Competition 2-Targets). The experiment took place at GIPSA-lab, Grenoble, 
    France, in 2015. A full description of the experiment is available at 
    https://hal.archives-ouvertes.fr/hal-02173913. Python code for manipulating 
    the data is available at https://github.com/plcrodrigues/py.BI.EEG.2015b-GIPSA. 
    The ID of this dataset is bi2015b.
    '''

    def __init__(self):
        self.subject_list = list(range(1, 22 + 1))
        self.pair_list = self.subject_list

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        chnames = ['FP1','FP2','AFz','F7','F3','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO7','O1','Oz','O2','PO8','PO9','PO10']
        chnames_subject1 = [chi + '_1' for chi in chnames]                                                      
        chnames_subject2 = [chi + '_2' for chi in chnames]      
        chnames = chnames_subject1 + chnames_subject2 + ['STI 014']
        chtypes = ['eeg'] * 64 + ['stim']               
        sessions = {}
        file_path_list = self.data_path(subject)        
        session_name_list = ['s1', 's2', 's3', 's4']
        for file_path, session_name in zip(file_path_list, session_name_list):

            sessions[session_name] = {}
            run_name = 'run_1'

            D = loadmat(file_path)['mat_data'].T

            S = D[1:65,:]
            stim = D[-1,:]
            idx_target = (stim >= 60) & (stim <=85)
            idx_nontarget = (stim >= 20) & (stim <=45)
            stim[idx_target] = 2
            stim[idx_nontarget] = 1

            X = np.concatenate([S, stim[None,:]])

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes, montage=None,
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            sessions[session_name][run_name] = raw

        return sessions

    # dummy function just for more readable code
    def _get_single_pair_data(self, pair):
        return self._get_single_subject_data(pair)          

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):

        if subject not in self.subject_list:
            raise(ValueError("Invalid subject number"))

        # check if has the .zip
        url = BI2015b_URL + 'group_' + str(subject).zfill(2) + '_mat.zip'
        path_zip = dl.data_path(url, 'BRAININVADERS2015B')
        path_folder = path_zip.strip('group_' + str(subject).zfill(2) + '_mat.zip')

        # check if has to unzip
        path_folder_subject = path_folder + 'group_' + str(subject).zfill(2) + os.sep
        if not(os.path.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            print('unzip', path_zip)
            zip_ref = zipfile.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []

        # filter the data regarding the experimental conditions
        pair_name = 'group_' + str(subject).zfill(2)
        subject_paths = [path_folder + pair_name + '/' + pair_name + '_s' + str(i) + '.mat' for i in range(1, 5)]

        return subject_paths
