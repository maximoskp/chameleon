#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:13:10 2018

@author: maximoskaliakatsos-papakostas
"""

import sys
import pickle
import os
cwd = os.getcwd()
sys.path.insert(0, cwd+'/CM_train')
sys.path.insert(0, cwd+'/CM_auxiliary')
import CM_TR_TrainingIdiom_class as tic
import Grouping_functions as grp
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

mainFolder = cwd + '/training_data/'
logging = True

for f in os.walk(mainFolder):
    if f[2][0].endswith('.xml'):
        print('f: ', f[0].split('/')[-1])
        # check if logging is required
        if logging:
            training_log_file = cwd+'/training_logs/'+f[0].split('/')[-1]
            prt.initialise_log_file( training_log_file )
        idiom = tic.TrainingIdiom(f[0]+'/', logging=logging, log_file=training_log_file)
        idiom = grp.group_gcts_of_idiom(idiom)
        # save learned idiom
        with open('trained_idioms/'+idiom.name+'.pickle', 'wb') as handle:
            pickle.dump(idiom, handle, protocol=pickle.HIGHEST_PROTOCOL)