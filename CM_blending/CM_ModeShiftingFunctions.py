#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:16:46 2018

@author: maximoskaliakatsos-papakostas
"""

import os
cwd = os.getcwd()
import numpy as np
import copy
import sys
sys.path.insert(0, cwd + '/CM_train')
import CM_TR_TrainingIdiom_class as tic
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf

def shift_gct_label(g, d):
    gnp = maf.str2np(g)
    gnp[0] = (gnp[0] + d)%12
    return maf.np2str(gnp.astype(int))
# end shift_gct_label

# CADENCES =================================================== CADENCES
def shift_cadence_label(c, d):
    # NEEDS str2np
    # split in hyphen
    c_tmp = c.split('-')
    c0 = shift_gct_label(c_tmp[0], d)
    c1 = shift_gct_label(c_tmp[1], d)
    return c0 + '-' + c1
# end shift_cadence_label
def shift_cadence_stucture(c, d):
    # final/penultimate gct labels
    c.final_gct = shift_gct_label(c.final_gct, d)
    c.penultimate_gct = shift_gct_label(c.penultimate_gct, d)
    # pair
    c.gcts_pair[0][0] = (c.gcts_pair[0][0]+d)%12
    c.gcts_pair[0].astype(int)
    c.gcts_pair[1][0] = (c.gcts_pair[1][0]+d)%12
    c.gcts_pair[1].astype(int)
    # relative pcp
    c.final_relative_pcp = np.roll( c.final_relative_pcp, d )
    c.penultimate_relative_pcp = np.roll( c.penultimate_relative_pcp, d )
    # label
    c.label = shift_cadence_label( c.label, d )
    return c
# end shift_cadence_stucture
def cadences_shift(cads, d):
    # counter is left unchanged
    for c_type in ['intermediate', 'final']:
        c = cads[c_type]
        # cadence ALL labels and structures - possibly not necessary
        for i in range(len(c.all_cadence_labels)):
            # all_labels
            c_lab = c.all_cadence_labels[i]
            cads[c_type].all_cadence_labels[i] = shift_cadence_label(c_lab, d)
            # all_structures
            c_struct = c.all_cadence_structures[i]
            cads[c_type].all_cadence_structures[i] = shift_cadence_stucture(c_struct, d)
        # simple labels, counter and dictionary/structures
        # keep dictionary keys
        all_keys = list( c.cadences_dictionary.keys() )
        print('all_keys: ', len(all_keys))
        print('cadence_labels: ', len(c.cadence_labels))
        for i in range( len( all_keys ) ):
            # labels
            c_lab = c.cadence_labels[i]
            cads[c_type].cadence_labels[i] = shift_cadence_label(c_lab, d)
            # dictionary key
            old_key = all_keys[i]
            new_key = shift_cadence_label(old_key, d)
            # counter key
            cads[c_type].cadences_counter[new_key] = cads[c_type].cadences_counter.pop(old_key)
            # dictionary key
            cads[c_type].cadences_dictionary[new_key] = cads[c_type].cadences_dictionary.pop(old_key)
            # it appears that it has already been done with the change of structures above
            # cads[c_type].cadences_dictionary[new_key] = shift_cadence_stucture( cads[c_type].cadences_dictionary[new_key], d )
    return cads
# end cadences_shift
# CADENCES =================================================== CADENCES

# GCT_INFO =================================================== GCT_INFO
def shift_gct_group_structures(s, d):
    # for each group structure
    for i in range( len( s ) ):
        # change member labels, np and rpcps
        for j in range( len( s[i].members ) ):
            s[i].members[j] = shift_gct_label( s[i].members[j], d )
            s[i].members_np[j][0] = (s[i].members_np[j][0] + d)%12
            s[i].members_rpcp[j] = np.roll( s[i].members_rpcp[j], d )
        # change representative label and np (no rpcp?)
        s[i].representative = shift_gct_label( s[i].representative, d )
        s[i].representative_np[0] = (s[i].representative_np[0] + d)%12
    return s
def gct_info_shift(g, d):
    # make a new info structure
    g_new = tic.GCT_info()
    # gct_group_structures
    g_new.gct_group_structures = shift_gct_group_structures(g.gct_group_structures, d)
    # create a new vl dictionary
    new_vl_dict = {}
    # gct vl dictionary keys
    all_keys = list(g.gct_vl_dict.keys())
    for i in range( len( all_keys ) ):
        old_key = all_keys[i]
        new_key = shift_gct_label( old_key, d )
        new_vl_dict[new_key] = g.gct_vl_dict.pop(old_key)
    g_new.gct_vl_dict = new_vl_dict
    # create a new membership dictionary
    new_membership_dict = {}
    # gcts_membership_dictionary need only key changes, contents changed with structures
    all_keys = list(g.gcts_membership_dictionary.keys())
    for i in range( len( all_keys ) ):
        old_key = all_keys[i]
        new_key = shift_gct_label( old_key, d )
        new_membership_dict[new_key] = g.gcts_membership_dictionary.pop(old_key)
    g_new.gcts_membership_dictionary = new_membership_dict
    # labels and rpcps
    for i in range( len( g.gcts_labels ) ):
        g_new.gcts_labels.append( shift_gct_label( g.gcts_labels[i], d ) )
        g_new.gcts_relative_pcs.append( np.roll( g.gcts_relative_pcs[i], d ) )
    # assign all others
    g_new.gcts_occurances = g.gcts_occurances
    g_new.gcts_probabilities = g.gcts_probabilities
    g_new.gcts_initial_probabilities = g.gcts_initial_probabilities
    g_new.gcts_markov = g.gcts_markov
    # did not assign: gcts_array, gcts_counter, gcts_initial_array, gcts_initial_counter, gcts_transitions_sum
    return g_new
# GCT_INFO =================================================== GCT_INFO
def parallel_shift_mode(m,d):
    print('parallel shifting')
    # change idiom name in mode
    m.idiom_name = m.idiom_name + '_D' + str(d)
    # cadences =================================================
    m.cadences = cadences_shift( m.cadences, d )
    # gct_group_info
    tmp_group_info = copy.deepcopy( m.gct_group_info )
    m.gct_group_info = gct_info_shift( tmp_group_info, d )
    # gct_info
    tmp_info = copy.deepcopy( m.gct_info )
    m.gct_info = gct_info_shift( tmp_info, d )
    # shift mode pcp
    m.mode_pcp = np.roll( m.mode_pcp, d )
    return m
# end parallel_shift_mode