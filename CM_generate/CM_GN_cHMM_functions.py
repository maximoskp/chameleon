#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:59:35 2018

@author: maximoskaliakatsos-papakostas
"""

import CM_similarity_functions as smf
import numpy as np
import copy
import CM_Misc_Aux_functions as maf
import os
cwd = os.getcwd()
import sys
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

def get_best_relpcp_matching_index(c, m):
    ' c is a relpcp chord '
    ' m is either a list of rpcp chords (if no groups are used) '
    ' or a list of lists of rpcp chords (lists of member pcps if groups are used) '
    best_corr = -1
    best_idx = 0
    for i in range( len(m) ):
        # check if it's rpcp directly
        if type( m[i] ) is np.ndarray:
            tmp_corr = np.corrcoef(c, m[i])[0,1]
            if best_corr < tmp_corr:
                best_corr = tmp_corr
                best_idx = i
        else:
            # if we have groups (list of lists) take each member's rpcp
            for j in range( len(m[i]) ):
                tmp_corr = np.corrcoef(c, m[i][j])[0,1]
                if best_corr < tmp_corr:
                    best_corr = tmp_corr
                    best_idx = i # keep the idx of the group
    return best_idx
# end get_best_relpcp_matching_index
def get_obs_probs(s, rpcps, members_probs):
    ' rpcps is either a list of rpcp chords (if no groups are used) '
    ' or a list of lists of rpcp chords (lists of member pcps if groups are used) '
    ' members_probs is ONLY used in case GCT GROUPING is used '
    p = np.zeros( (len(rpcps) , len(s)) )
    for i in range( len(s) ):
        for j in range( len(rpcps) ):
            # check if it's an np array or a list
            rpcp = rpcps[j]
            mem_probs = members_probs[j]
            if type( rpcp ) is np.ndarray:
                p[j,i] = smf.evaluate_melody_chord_rpcp_matching(s[i].relative_pcp, s[i].important_relative_pcp, rpcps[j])
            else:
                # get all members' rpcps
                tmp_max_match = -10000.0
                for ii in range( len( rpcp ) ):
                    mmbr = rpcp[ii]
                    tmp_match = smf.evaluate_melody_chord_rpcp_matching(s[i].relative_pcp, s[i].important_relative_pcp, mmbr)
                    # multiply the probability of member occurance
                    # to eliminate rare members that happen to have the note as extension
                    tmp_match = tmp_match*np.power(mem_probs[ii], 20)
                    if tmp_max_match < tmp_match:
                        tmp_max_match = tmp_match
                p[j,i] = tmp_max_match
    # # add a small value to zeros
    p[ p == 0 ] = 0.0000000001
    # make distribution column-wise
    for j in range( p.shape[1] ):
        if np.sum( p[:,j] ) > 0:
            p[:,j] = p[:,j]/np.sum(p[:,j])
    return p
# end get_obs_probs
def apply_cHMM_with_constraints(obs, c, seg_idxs, sts_idxs):
    # adventure exponent
    adv_exp = 0.7
    markov = copy.deepcopy( c.gcts_markov )
    # smooth markov
    markov[ markov == 0 ] = 0.00000001
    # neutralise diagonal
    for i in range(markov.shape[0]):
        markov[i,i] = 0.000000001*markov[i,i]
    # apply adventure
    markov = np.power(markov, adv_exp)
    # re-normalise
    for i in range(markov.shape[0]):
        if np.sum( markov[i,:] ) > 0:
            markov[i,:] = markov[i,:]/np.sum( markov[i,:] )
    # beginning chord probabilities
    pr = c.gcts_initial_probabilities
    # cads = np.ones(markov.shape[0])
    delta = np.zeros( ( markov.shape[0] , obs.shape[1]) )
    psi = np.zeros( ( markov.shape[0] , obs.shape[1]) )
    pathIDXs = np.zeros( obs.shape[1] )
    t = 0
    if sts_idxs[0] != -1:
        delta[:,t] = np.zeros( markov.shape[0] )
        delta[ sts_idxs[0] ] = 1
    else:
        delta[:,t] = np.multiply( pr , obs[:,t] )
        if np.sum(delta[:,t]) != 0:
            delta[:,t] = delta[:,t]/np.sum(delta[:,t])
    
    psi[:,t] = 0 # arbitrary value, since there is no predecessor to t=0
    
    for t in range(1, obs.shape[1], 1):
        if (t != obs.shape[1]-1) or (sts_idxs[1] == -1) :
            for j in range(0, markov.shape[0]):
                tmp_trans_prob = markov[:,j]
                # if np.sum( tmp_trans_prob ) != 0:
                #     tmp_trans_prob = tmp_trans_prob/np.sum( tmp_trans_prob )
                delta[j,t] = np.max( np.multiply(delta[:,t-1], tmp_trans_prob)*obs[j,t] )
                psi[j,t] = np.argmax( np.multiply(delta[:,t-1], tmp_trans_prob)*obs[j,t] )
            if np.sum(delta[:,t]) != 0:
                delta[:,t] = delta[:,t]/np.sum(delta[:,t])
        else:
            j = sts_idxs[1]
            tmp_trans_prob = markov[:,j]
            # if np.sum( tmp_trans_prob ) != 0:
            #     tmp_trans_prob = tmp_trans_prob/np.sum( tmp_trans_prob )
            delta[j,t] = np.max( np.multiply(delta[:,t-1], tmp_trans_prob)*obs[j,t] )
            psi[j,t] = np.argmax( np.multiply(delta[:,t-1], tmp_trans_prob)*obs[j,t] )
            if np.sum(delta[:,t]) != 0:
                delta[:,t] = delta[:,t]/np.sum(delta[:,t])
    # end for t
    # print('delta: ', delta)
    if sts_idxs[1] == -1:
        pathIDXs[obs.shape[1]-1] = int(np.argmax(delta[:,obs.shape[1]-1]))
    else:
        pathIDXs[obs.shape[1]-1] = int(sts_idxs[1])
    
    for t in range(obs.shape[1]-2, -1, -1):
        pathIDXs[t] = int(psi[ int(pathIDXs[t+1]) , t+1 ])
    print('pathIDXs: ', pathIDXs)
    gcts_out = []
    gct_labels_out = []
    for i in range( len(pathIDXs) ):
        gcts_out.append( maf.str2np(c.gcts_labels[ int(pathIDXs[i]) ]) )
        gct_labels_out.append( c.gcts_labels[ int(pathIDXs[i]) ] )
    return gcts_out, gct_labels_out, delta, psi
# end apply_cHMM_with_constraints
def apply_cHMM_to_segments(p,c, seg_idxs, sts_idxs, m, logging=False):
    # m is only for getting file name for logging
    # for every segment
    for i in range(1, len( seg_idxs ), 1):
        # check if the length of the segment is > 1
        if seg_idxs[i] - seg_idxs[i-1] > 0:
            # get observation probabilities
            obs_probs = get_obs_probs( p.melody_chord_segments[seg_idxs[i-1]:(seg_idxs[i]+1)], c.gcts_relative_pcs, c.gcts_probabilities )
            # log - image obs_probs matrix
            if logging:
                # first make x and y labels
                # y_labels are chords
                y_labels = []
                for ii in range( len( c.gcts_labels ) ):
                    y_labels.append( str(ii) + '-' + c.gcts_labels[ii] )
                # x_labels are chord segment melody midis
                x_labels = []
                for cs in p.melody_chord_segments[seg_idxs[i-1]:(seg_idxs[i]+1)]:
                    x_labels.append( maf.np2str( np.where( cs.relative_pcp > 0 )[0] ) + '-i' + maf.np2str( np.where( cs.important_relative_pcp > 0 )[0] ) )
                # print observations
                prt.print_image_with_axis(m.harmonisation_file_name+'_obs'+'_'+str(p.index)+'_'+str(seg_idxs[i-1]), obs_probs, x_labels, y_labels, vertical_x=True)
                # print markov
                prt.print_image_with_axis(m.harmonisation_file_name+'_markov'+'_'+str(p.index)+'_'+str(seg_idxs[i-1]), c.gcts_markov, y_labels, y_labels, vertical_x=True)
            gcts_out, gct_labels_out, delta, psi = apply_cHMM_with_constraints(obs_probs, c, seg_idxs[ (i-1):(i+1) ], sts_idxs[ (i-1):(i+1) ])
            # log - image delta trellis matrix
            if logging:
                # first make x and y labels
                # y_labels are chords
                y_labels = []
                for ii in range( len( c.gcts_labels ) ):
                    y_labels.append( str(ii) + '-' + c.gcts_labels[ii] )
                # x_labels are chord segment melody midis
                x_labels = []
                for cs in p.melody_chord_segments[seg_idxs[i-1]:(seg_idxs[i]+1)]:
                    x_labels.append( maf.np2str( np.where( cs.relative_pcp > 0 )[0] ) + '-i' + maf.np2str( np.where( cs.important_relative_pcp > 0 )[0] ) )
                prt.print_image_and_numbers_with_axis(m.harmonisation_file_name+'_trellis'+'_'+str(p.index)+'_'+str(seg_idxs[i-1]), delta, psi.astype(int), x_labels, y_labels, vertical_x=True)
            # apply gcts to phrase
            for j in range(seg_idxs[i-1], seg_idxs[i], 1):
                p.melody_chord_segments[j].gct_chord = gcts_out[j - seg_idxs[i-1]]
                p.melody_chord_segments[j].gct_label = gct_labels_out[j - seg_idxs[i-1]]
    return p
# end apply_cHMM_to_segments
def apply_cHMM_to_phrase_from_mode(p, c, m, logging=False):
    # run through phrase and isolate starting, ending and constraint melody
    # chord segment indexes of constraints along with the state index
    # m is only for logging, to get the file name
    segment_indexes = []
    state_indexes = []
    for i in range( len( p.melody_chord_segments ) ):
        # log chord segment info
        if logging:
            tmp_log_line = 'chord segment info:' + '\n'
            tmp_log_line += 'MIDI notes: ' + str(p.melody_chord_segments[i].melody_midi) + '\n'
            tmp_log_line += 'Important MIDI notes: ' + str(p.melody_chord_segments[i].important_melody_midi) + '\n'
            tmp_log_line += 'Relative pcp: ' + str(p.melody_chord_segments[i].relative_pcp) + '\n'
            tmp_log_line += 'Important relative pcp: ' + str(p.melody_chord_segments[i].important_relative_pcp) + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        if p.melody_chord_segments[i].is_constraint:
            segment_indexes.append(i)
            state_indexes.append( get_best_relpcp_matching_index( p.melody_chord_segments[i].gct_rpcp, c.gcts_relative_pcs) )
            # log chord segment info
            if logging:
                tmp_log_line = 'chord is constraint:' + '\n'
                tmp_log_line += 'User-given constraint: ' + str(p.melody_chord_segments[i].gct_rpcp) + '\n'
                tmp_log_line += 'Found as corresponding with idiom chord: ' + c.gcts_labels[ state_indexes[-1] ]
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # after finding the constraints, check if the first and last are included
    # else include them
    if 0 not in segment_indexes:
        segment_indexes = [0] + segment_indexes
        state_indexes = [-1] + state_indexes
    if (len( p.melody_chord_segments ) - 1) not in segment_indexes:
        segment_indexes.append( len( p.melody_chord_segments ) - 1 )
        state_indexes.append( -1 )
    # apply cHMM to given areas and constraints
    p = apply_cHMM_to_segments(p, c, segment_indexes, state_indexes, m, logging=logging)
    return p
#end apply_cHMM_to_phrase_from_mode
def apply_cHMM_to_melody_from_idiom(m,idiom, use_GCT_grouping, logging=False):
    print('applying cHMM')
    # log that it's about cHMM
    if logging:
        tmp_log_line = 'cHMM =========================================== ' + '\n'
        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # run through all phrases
    for p in m.phrases:
        # get proper mode that corresponds to the mode of the phrase
        mode = smf.get_best_matching_mode( p.tonality.mode_pcp, idiom )
        # log selected mode
        if logging:
            tmp_log_line = 'NEW PHRASE ========================= ' + '\n'
            tmp_log_line += 'User-given phrase mode: ' + str(p.tonality.mode_pcp) + '\n'
            tmp_log_line += 'cHMM selected mode: ' + mode.mode_name
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        p.idiom_mode_label = mode.mode_name
        # decide if grouping is used
        chord_info = mode.gct_info
        if use_GCT_grouping:
            chord_info = mode.gct_group_info
            # log grouping
            if logging:
                tmp_log_line = ' - Using grouping'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # apply cHMM of mode
        p = apply_cHMM_to_phrase_from_mode( p, chord_info, m, logging=logging )
    return m
# end apply_cadences_to_melody_from_idiom