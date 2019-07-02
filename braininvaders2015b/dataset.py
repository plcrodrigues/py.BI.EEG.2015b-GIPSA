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

BI2014b_URL = 'https://zenodo.org/record/XXXXXX/files/'

class BrainInvaders2015b():
    '''

    '''

    def __init__(self):
        self.subject_list = list(range(1, 19 + 1))
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

        # if subject not in self.subject_list:
        #     raise(ValueError("Invalid subject number"))

        # # check if has the .zip
        # url = BI2015b_URL + 'subject_' + str(subject).zfill(2) + '.zip'
        # path_zip = dl.data_path(url, 'BRAININVADERS2015B')
        # path_folder = path_zip.strip('subject_' + str(subject).zfill(2) + '.zip')

        # # check if has to unzip
        # path_folder_subject = path_folder + 'subject_' + str(subject).zfill(2) + os.sep
        # if not(os.path.isdir(path_folder_subject)):
        #     os.mkdir(path_folder_subject)
        #     print('unzip', path_zip)
        #     zip_ref = zipfile.ZipFile(path_zip, "r")
        #     zip_ref.extractall(path_folder_subject)

        # subject_paths = []

        # # filter the data regarding the experimental conditions
        # subject_paths.append(path_folder_subject + 'subject_' + str(subject).zfill(2) + '.mat')        
        pair_name = 'group_' + str(subject).zfill(2)
        file_base = '/localdata/coelhorp/Datasets/gregoire/Final/2015/Group_B/mat/'
        subject_paths = [file_base + pair_name + '_mat/' + pair_name + '_s' + str(i) + '.mat' for i in range(1, 5)]

        return subject_paths
