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
sys.path.insert(0, cwd + '/CM_generate')
sys.path.insert(0, cwd + '/CM_NN_VL')
import CM_GN_voice_leading_functions as vlf

# CADENCES =================================================== CADENCES
def merge_cadences(idiom_name, mode_name, cads1, cads2):
    # construct empty objects for intermediate and final cadences
    c_i = tic.TrainingCadences( idiom_name, mode_name, 'intermediate' )
    c_f = tic.TrainingCadences( idiom_name, mode_name, 'final' )
    # start merging
    for c_type in ['intermediate', 'final']:
        c1 = cads1[c_type]
        c2 = cads2[c_type]
        if c_type == 'intermediate':
            # cadence ALL labels and structures - possibly not necessary
            # all_labels
            c_i.all_cadence_labels.extend( c1.all_cadence_labels )
            c_i.all_cadence_labels.extend( c2.all_cadence_labels )
            # all_structures
            c_i.all_cadence_structures.extend( c1.all_cadence_structures )
            c_i.all_cadence_structures.extend( c2.all_cadence_structures )
            # remake cadence stats
            c_i.make_cadences_stats()
        else:
            # cadence ALL labels and structures - possibly not necessary
            # all_labels
            c_f.all_cadence_labels.extend( c1.all_cadence_labels )
            c_f.all_cadence_labels.extend( c2.all_cadence_labels )
            # all_structures
            c_f.all_cadence_structures.extend( c1.all_cadence_structures )
            c_f.all_cadence_structures.extend( c2.all_cadence_structures )
            # remake cadence stats
            c_f.make_cadences_stats()
    cads = {'intermediate': c_i, 'final': c_f}
    return cads
# end cadences_shift
# CADENCES =================================================== CADENCES

# GCT_INFO =================================================== GCT_INFO
def make_extended_makov(m1, m2, l1, l2):
    # construct expanded with zeros
    m = np.vstack( (m1 , np.zeros( (m2.shape[0],m1.shape[1]) )) )
    m = np.hstack( (m , np.vstack( (np.zeros( (m1.shape[0], m2.shape[1]) ), m2) )) )
    # find common chords based on labels
    for i in range( len(l1) ):
        for j in range( len(l2) ):
            if l1[i] == l2[j]: # TODO: check if they belong to the same group
                # horizontal bottom-right to top-right
                m[ i , len(l1): ] = m[ len(l1)+j , len(l1): ]
                # horizontal top-left to bottom-left
                m[ len(l1)+j , :len(l1) ] = m[ i , :len(l1) ]
                # vertical top-left to top-right
                m[ :len(l1) , i ] = m[ :len(l1) , len(l1)+j ]
                # vertical bottom-left to bottom-right
                m[ len(l1): , len(l1)+j ] = m[ len(l1): , i ]
    # re-normalise matrix
    for i in range( m.shape[0] ):
        if np.sum( m[i,:] ) != 0:
            m[i,:] = m[i,:]/np.sum( m[i,:] )
    return m
# end make_extended_makov
def merge_gct_info(g1, g2):
    # initialise empty gct_info object
    g = tic.GCT_info()
    # merge gcts_array
    g.gcts_array.extend( g1.gcts_array )
    g.gcts_array.extend( g2.gcts_array )
    # gcts_counter is NOT merged - possibly we don't need it
    # merge gct_group_structures
    g.gct_group_structures.extend( g1.gct_group_structures )
    g.gct_group_structures.extend( g2.gct_group_structures )
    # gcts_membership_dictionary is not merged - posibly we don't need it
    # merge gcts_labels
    g.gcts_labels.extend( g1.gcts_labels )
    g.gcts_labels.extend( g2.gcts_labels )
    # merge gcts_relative_pcs
    g.gcts_relative_pcs.extend( g1.gcts_relative_pcs )
    g.gcts_relative_pcs.extend( g2.gcts_relative_pcs )
    # merge gcts_occurances
    g.gcts_occurances.extend( g1.gcts_occurances )
    g.gcts_occurances.extend( g2.gcts_occurances )
    # merge gcts_probabilities
    g.gcts_probabilities.extend( g1.gcts_probabilities )
    g.gcts_probabilities.extend( g2.gcts_probabilities )
    # merge gcts_initial_array
    g.gcts_initial_array.extend( g1.gcts_initial_array )
    g.gcts_initial_array.extend( g2.gcts_initial_array )
    # gcts_initial_counter is not merged - possible not necessary
    # merge gcts_initial_probabilities
    g.gcts_initial_probabilities.extend( g1.gcts_initial_probabilities )
    g.gcts_initial_probabilities.extend( g2.gcts_initial_probabilities )
    # gcts_transitions_sum is not merged
    g.gcts_markov = make_extended_makov( g1.gcts_markov , g2.gcts_markov , g1.gcts_labels , g2.gcts_labels )
    # merge gct_vl_dict
    g.gct_vl_dict = {}
    # get list of keys/gcts from idiom1 and idiom2
    k1 = list( g1.gct_vl_dict.keys() )
    k2 = list( g2.gct_vl_dict.keys() )
    # run through all k1 labels and if a label belonging to k2 is found, "blend"
    for k in k1:
        tmp_vl_1 = copy.deepcopy(g1.gct_vl_dict[ k ])
        if k in k2:
            tmp_vl_2 = g2.gct_vl_dict[ k ]
            for l in ['inversions', 'mel2bass', 'to_bvl', 'from_bvl']:
                tmp_vl_1[l] = tmp_vl_1[l]/2 + tmp_vl_2[l]/2
        g.gct_vl_dict[ k ] = tmp_vl_1
    # it remains to find k2 elements that have not yet been incorporated
    for k in k2:
        tmp_vl_2 = copy.deepcopy(g2.gct_vl_dict[ k ])
        # if shown earlier, it should not be included
        if k not in k1:
            g.gct_vl_dict[ k ] = tmp_vl_2
    # merge gct_vl - CAUTION: the order of vl objects in this array should be the same as gcts_labels
    g.gct_vl = []
    for l in g.gcts_labels:
        g.gct_vl.append( g.gct_vl_dict[ l ] )
    return g
# GCT_INFO =================================================== GCT_INFO

def integrate_partial_blends( g, b, groups ):
    gct_labels = g.gcts_labels
    # run through each A->B blend pair
    for t_pair in b:
        # get both transitions in pair/chain
        t1 = t_pair[0]
        t2 = t_pair[1]
        # print('t1: ', t1.from_chord_np['property'], ' - ', t1.to_chord_np['property'])
        # print('t2: ', t2.from_chord_np['property'], ' - ', t2.to_chord_np['property'])
        # check if any chord matches chords already in idiom
        # T1 ===========================================================
        t1_from_is_new = False
        t1_to_is_new = False
        if maf.np2str( t1.from_chord_np['property'].astype(int) ) in gct_labels:
            t1_idx1 = gct_labels.index( maf.np2str( t1.from_chord_np['property'].astype(int) ) )
        else:
            t1_from_is_new = True
            # consider an index that extends the matrix by 1 element
            t1_idx1 = g.gcts_markov.shape[0]
            # append new gcts to labels
            tmp_label = maf.np2str( t1.from_chord_np['property'].astype(int) )
            g.gcts_labels.append( tmp_label )
            # to append to vl dictionary
            g.gct_vl_dict[ tmp_label ] = vlf.make_neutral_bbvl( t1.from_chord_np['property'] )
            g.gct_vl.append( g.gct_vl_dict[ tmp_label ] )
            # rpcp of new chord
            if groups:
                g.gcts_relative_pcs.append( [ maf.gct2relpcp( t1.from_chord_np['property'] ) ] )
            else:
                g.gcts_relative_pcs.append( maf.gct2relpcp( t1.from_chord_np['property'] ) )
            # probability of new chord
            if groups:
                g.gcts_probabilities.append( [0.5] )
            else:
                g.gcts_probabilities.append( 0.5 )
            # initial probabilities
            g.gcts_initial_probabilities.append( 0.1 )
        if maf.np2str( t1.to_chord_np['property'].astype(int) ) in gct_labels:
            t1_idx2 = gct_labels.index( maf.np2str( t1.to_chord_np['property'].astype(int) ) )
        else:
            t1_to_is_new = True
            # consider an index that extends the matrix by 1 element
            t1_idx2 = g.gcts_markov.shape[0]
            # append new gcts to labels
            tmp_label = maf.np2str( t1.to_chord_np['property'].astype(int) )
            g.gcts_labels.append( tmp_label )
            # to append to vl dictionary
            g.gct_vl_dict[ tmp_label ] = vlf.make_neutral_bbvl( t1.to_chord_np['property'] )
            g.gct_vl.append( g.gct_vl_dict[ tmp_label ] )
            # rpcp of new chord
            if groups:
                g.gcts_relative_pcs.append( [ maf.gct2relpcp( t1.to_chord_np['property'] ) ] )
            else:
                g.gcts_relative_pcs.append( maf.gct2relpcp( t1.to_chord_np['property'] ) )
            # probability of new chord
            if groups:
                g.gcts_probabilities.append( [0.5] )
            else:
                g.gcts_probabilities.append( 0.5 )
            # initial probabilities
            g.gcts_initial_probabilities.append( 0.1 )
        # T2 ===========================================================
        t2_from_is_new = False
        t2_to_is_new = False
        if maf.np2str( t2.from_chord_np['property'].astype(int) ) in gct_labels:
            t2_idx1 = gct_labels.index( maf.np2str( t2.from_chord_np['property'].astype(int) ) )
        else:
            t2_from_is_new = True
            # consider an index that extends the matrix by 1 element
            t2_idx1 = g.gcts_markov.shape[0]
            # append new gcts to labels
            tmp_label = maf.np2str( t2.from_chord_np['property'].astype(int) )
            g.gcts_labels.append( tmp_label )
            # to append to vl dictionary
            g.gct_vl_dict[ tmp_label ] = vlf.make_neutral_bbvl( t2.from_chord_np['property'] )
            g.gct_vl.append( g.gct_vl_dict[ tmp_label ] )
            # rpcp of new chord
            if groups:
                g.gcts_relative_pcs.append( [ maf.gct2relpcp( t2.from_chord_np['property'] ) ] )
            else:
                g.gcts_relative_pcs.append( maf.gct2relpcp( t2.from_chord_np['property'] ) )
            # probability of new chord
            if groups:
                g.gcts_probabilities.append( [0.5] )
            else:
                g.gcts_probabilities.append( 0.5 )
            # initial probabilities
            g.gcts_initial_probabilities.append( 0.1 )
        if maf.np2str( t2.to_chord_np['property'].astype(int) ) in gct_labels:
            t2_idx2 = gct_labels.index( maf.np2str( t2.to_chord_np['property'].astype(int) ) )
        else:
            t2_to_is_new = True
            # consider an index that extends the matrix by 1 element
            t2_idx2 = g.gcts_markov.shape[0]
            # append new gcts to labels
            tmp_label = maf.np2str( t2.to_chord_np['property'].astype(int) )
            g.gcts_labels.append( tmp_label )
            # to append to vl dictionary
            g.gct_vl_dict[ tmp_label ] = vlf.make_neutral_bbvl( t2.to_chord_np['property'] )
            g.gct_vl.append( g.gct_vl_dict[ tmp_label ] )
            # rpcp of new chord
            if groups:
                g.gcts_relative_pcs.append( [ maf.gct2relpcp( t2.to_chord_np['property'] ) ] )
            else:
                g.gcts_relative_pcs.append( maf.gct2relpcp( t2.to_chord_np['property'] ) )
            # probability of new chord
            if groups:
                g.gcts_probabilities.append( [0.5] )
            else:
                g.gcts_probabilities.append( 0.5 )
            # initial probabilities
            g.gcts_initial_probabilities.append( 0.1 )
        # add each chain in the matrix
        # T1 ===== check if new rows need to be added
        if t1_from_is_new:
            # add new row below the matrix
            g.gcts_markov = np.vstack( (g.gcts_markov, np.zeros( (1,g.gcts_markov.shape[1]) )) )
            # add new column to the right of the matrix
            g.gcts_markov = np.hstack( (g.gcts_markov, np.zeros( (g.gcts_markov.shape[0],1) )) )
        if t1_to_is_new:
            # add new row below the matrix
            g.gcts_markov = np.vstack( (g.gcts_markov, np.zeros( (1,g.gcts_markov.shape[1]) )) )
            # add new column to the right of the matrix
            g.gcts_markov = np.hstack( (g.gcts_markov, np.zeros( (g.gcts_markov.shape[0],1) )) )
        # T2 ===== check if new rows need to be added
        if t2_from_is_new:
            # add new row below the matrix
            g.gcts_markov = np.vstack( (g.gcts_markov, np.zeros( (1,g.gcts_markov.shape[1]) )) )
            # add new column to the right of the matrix
            g.gcts_markov = np.hstack( (g.gcts_markov, np.zeros( (g.gcts_markov.shape[0],1) )) )
        if t2_to_is_new:
            # add new row below the matrix
            g.gcts_markov = np.vstack( (g.gcts_markov, np.zeros( (1,g.gcts_markov.shape[1]) )) )
            # add new column to the right of the matrix
            g.gcts_markov = np.hstack( (g.gcts_markov, np.zeros( (g.gcts_markov.shape[0],1) )) )
        # activate matrix values
        g.gcts_markov[t1_idx1, t1_idx2] = 0.5
        g.gcts_markov[t2_idx1, t2_idx2] = 0.5
        # append new gcts to group structures with members only themselves, regardless of the whether they do belong in other groups (or other groups belong to them)
    return g
# end integrate_partial_blends

def integrate_blends( g , ab_blends, ba_blends, groups=False ):
    print('integrating blends')
    g = integrate_partial_blends( g, ab_blends, groups )
    g = integrate_partial_blends( g, ba_blends, groups )
    return g
# end integrate_blends

def compile_blended_idiom(m1, m2, ton_diff, best_ab_blends, best_ba_blends):
    print('compiling blended idiom')
    # define blended idiom name
    idiom_name = 'bl_' + m1.idiom_name + m1.mode_name + '_D' + str(ton_diff) + '_' + m2.idiom_name + m2.mode_name
    # construct idiom
    idiom = tic.BlendingIdiom( idiom_name )
    # construct mode - start with name from mode pcps
    m1_pcs = np.where( m1.mode_pcp > 0 )
    m2_pcs = np.where( m2.mode_pcp > 0 )
    m_pcs = np.union1d( m1_pcs , m2_pcs )
    mode_name = maf.np2str( m_pcs )
    tmp_mode = tic.TrainingMode( idiom_name, m_pcs )
    idiom.modes[mode_name] = tmp_mode
    # cadences
    idiom.modes[mode_name].cadences = merge_cadences( idiom_name, mode_name, m1.cadences, m2.cadences )
    # gct_info
    idiom.modes[mode_name].gct_info = merge_gct_info( m1.gct_info, m2.gct_info )
    # gct_group_info
    idiom.modes[mode_name].gct_group_info = merge_gct_info( m1.gct_group_info, m2.gct_group_info )
    # integrate blended chords/transitions in the expanded matrix
    idiom.modes[mode_name].gct_info = integrate_blends( idiom.modes[mode_name].gct_info , best_ab_blends, best_ba_blends, groups=False )
    idiom.modes[mode_name].gct_group_info = integrate_blends( idiom.modes[mode_name].gct_info , best_ab_blends, best_ba_blends, groups=True )
    return idiom
# end compile_blended_idiom