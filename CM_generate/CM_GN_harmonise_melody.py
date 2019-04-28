#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 07:32:52 2018

@author: maximoskaliakatsos-papakostas
"""

import pickle
import CM_GN_MelodyInput_class as mld
import CM_GN_cadence_functions as cdn
import CM_GN_cHMM_functions as chmm
import CM_GN_voice_leading_functions as vlf
import CM_user_output_functions as uof
import os
cwd = os.getcwd()
import sys
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

def harmonise_melody_with_idiom(melodyFolder, melodyFileName, idiomName, targetFolder='server_harmonised_output/', mode_in='Auto', use_GCT_grouping=False, voice_leading='simple', logging=False):
    '''
    voice_leading: 'simple', 'nn', 'bidirectional_bvl', TODO: 'markov_bvl'
    '''
    # construct melody structure
    m = mld.MelodyInput(melodyFolder, melodyFileName)
    # keep name for logging
    m.harmonisation_file_name = targetFolder+m.name+'_'+idiomName
    # check if logging is required
    if logging:
        prt.initialise_log_file( m.harmonisation_file_name )
    # load idiom
    idiomFolder = cwd
    with open(idiomFolder+'/trained_idioms/'+idiomName+'.pickle', 'rb') as handle:
        idiom = pickle.load(handle)
    # apply cadences to phrases
    m = cdn.apply_cadences_to_melody_from_idiom(m, idiom ,mode_in=mode_in, logging=logging)
    # apply cHMM
    m = chmm.apply_cHMM_to_melody_from_idiom(m, idiom, use_GCT_grouping, mode_in=mode_in, logging=logging)
    # apply voice leading
    if voice_leading is 'simple':
        m = vlf.apply_simple_voice_leading(m, idiom)
    elif voice_leading is 'nn':
        m = vlf.apply_NN_voice_leading(m, idiom)
    elif voice_leading is 'bidirectional_bvl':
        bbvl = vlf.BBVL(m, idiom, use_GCT_grouping, mode_in=mode_in)
        m = bbvl.apply_bbvl()
    # export to desired format
    uof.generate_xml(m.output_stream, fileName=m.harmonisation_file_name+'.xml')
    return m, idiom