#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:04:32 2018

@author: maximoskaliakatsos-papakostas

"""

'''
Functions for:
- getting the best matching training mode, given an input mode pcp
'''

import numpy as np
import copy

def get_best_matching_mode(mode_pcp_in, idiom):
    ' receives a mode_pcp_in mode pcp array and an idiom structure '
    ' and returns the idiom mode that best matches the correlation '
    ' of the input pcp array '
    max_corr = -1
    m_out = []
    # for all modes, compare correlations
    for m in idiom.modes.values():
        if max_corr < np.corrcoef( mode_pcp_in, m.mode_pcp )[0,1]:
            max_corr = np.corrcoef( mode_pcp_in, m.mode_pcp )[0,1]
            m_out = copy.deepcopy(m)
    return m_out
# end get_best_matching_mode
def evaluate_melody_chord_rpcp_matching(n, m, c):
    # n: relpcp of all notes in the segment
    # m: >> important notes >>
    # c: the chord relpcp
    p = 0
    # first gather positives - melody notes found in chords
    for i in range(len( n )):
        # if important melody note in gct, add big reward
        if m[i] >= 1 and c[i] >= 1:
            p += 0.9
        elif n[i] >= 1 and c[i] >= 1:
            p += 0.1
    # normalise according to rpcs in melody
    if np.count_nonzero( n ) > 0:
        p = p/( 0.1*np.count_nonzero( n ) + 0.9*np.count_nonzero( m ) )
    # # then penalise inexistances
    # for i in range(len( n )):
    #     # if important melody note NOT in gct, penalise big time
    #     if m[i] == 1 and c[i] == 0:
    #         p = p/10000000000
    #     elif n[i] == 1 and c[i] == 0:
    #         p = p/100000
    return p
# end evaluate_melody_chord_rpcp_matching
def evaluate_cadence_in_phrase(p, c):
    # find best score for final and penultimate chords
    final_score = 1
    penultimate_score = 1
    # if final chord is not constraint
    if not p.melody_chord_segments[-1].is_constraint:
        final_score = evaluate_melody_chord_rpcp_matching(p.melody_chord_segments[-1].relative_pcp,  p.melody_chord_segments[-1].important_relative_pcp, c.final_relative_pcp )
    # check if penultimate chord exists and if it is constraint
    if len(p.melody_chord_segments) < 2:
        penultimate_score = 1
    else:
        if not p.melody_chord_segments[-2].is_constraint:
            penultimate_score = evaluate_melody_chord_rpcp_matching(p.melody_chord_segments[-2].relative_pcp,  p.melody_chord_segments[-2].important_relative_pcp, c.penultimate_relative_pcp )
    # cadence_score = (0.8*final_score + 0.2*penultimate_score)*c.probability
    cadence_score = (0.5*final_score + 0.5*penultimate_score)
    # cadence_score = np.power(final_score, 1)*np.power(penultimate_score, 3)*c.probability
    if np.isnan(cadence_score):
        cadence_score = 0.0
    return cadence_score
# end evaluate_cadence_in_phrase
def get_best_matching_cadence(p, m):
    ' gets phrase structure from melody p and mode from learned idiom m '
    ' and returns the best matching cadence object '
    c_out = []
    overall_score = -1
    # check if phrase has no chords
    if len(p.melody_chord_segments) < 1:
        print('WEIRD melody input in CM_similarity_functions.py: phrase has no chord segments')
    else: # if it does have at least one chord
        # check if phrase is final or intermediate
        all_cads = m.cadences['intermediate']
        other_cads = m.cadences['final']
        # check if it's a 'final' cadence or if list of intermediate cadences is empty
        if p.level > 1 or len( all_cads.cadences_dictionary.values() ) == 0:
            all_cads = m.cadences['final']
            other_cads = m.cadences['intermediate']
        # run through all cadences and evaluate them
        for c in all_cads.cadences_dictionary.values():
            tmp_score = evaluate_cadence_in_phrase( p, c )
            if overall_score < tmp_score:
                overall_score = tmp_score
                c_out = c
        # check if score too low and use alternative cadences
        if overall_score < 0.7:
            # run throuth all alternative cadences and evaluate them
            for c in other_cads.cadences_dictionary.values():
                tmp_score = evaluate_cadence_in_phrase( p, c )
                if overall_score < tmp_score:
                    overall_score = tmp_score
                    c_out = c
    return c_out
# end get_best_matching_cadence