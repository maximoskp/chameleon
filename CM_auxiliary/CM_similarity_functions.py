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
import os
cwd = os.getcwd()
import sys
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

def get_best_matching_mode(mode_pcp_in, idiom):
    ' receives a mode_pcp_in mode pcp array and an idiom structure '
    ' and returns the idiom mode that best matches the correlation '
    ' of the input pcp array '
    max_corr = -2
    m_out = []
    # for all modes, compare correlations
    for m in idiom.modes.values():
        # check if given mode or idiom mode are chromatic (all 1s or 0s)
        if np.all( mode_pcp_in == mode_pcp_in[0] ) or np.all( m.mode_pcp == m.mode_pcp[0] ):
            tmp_corr = -1
        else:
            tmp_corr = np.corrcoef( mode_pcp_in, m.mode_pcp )[0,1]
        if np.isnan(tmp_corr):
            tmp_corr = -1
        if max_corr < tmp_corr:
            max_corr = tmp_corr
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
def evaluate_cadence_in_phrase(p, c, m, logging=False):
    # find best score for final and penultimate chords
    final_score = 1
    penultimate_score = 1
    # if final chord is not constraint
    if not p.melody_chord_segments[-1].is_constraint:
        final_score = evaluate_melody_chord_rpcp_matching(p.melody_chord_segments[-1].relative_pcp,  p.melody_chord_segments[-1].important_relative_pcp, c.final_relative_pcp )
        # log eval final chord
        if logging:
            # log final score
            tmp_log_line = 'Cadence final chord: ' + c.final_gct + '\n'
            tmp_log_line += 'cadence final rpcp: ' + str(c.final_relative_pcp) + '\n'
            tmp_log_line += 'melody final rpcp: ' + str(p.melody_chord_segments[-1].relative_pcp) + '\n'
            tmp_log_line += 'melody important final rpcp: ' + str(p.melody_chord_segments[-1].important_relative_pcp) + '\n'
            tmp_log_line += 'Final chord score: ' + "{:.4f}".format(final_score)
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # check if penultimate chord exists and if it is constraint
    if len(p.melody_chord_segments) < 2:
        penultimate_score = 1
    else:
        if not p.melody_chord_segments[-2].is_constraint:
            penultimate_score = evaluate_melody_chord_rpcp_matching(p.melody_chord_segments[-2].relative_pcp,  p.melody_chord_segments[-2].important_relative_pcp, c.penultimate_relative_pcp )
            # log eval penultimate chord
            if logging:
                # log penultimate score
                tmp_log_line = 'Cadence penultimate chord: ' + c.penultimate_gct + '\n'
                tmp_log_line += 'cadence penultimate rpcp: ' + str(c.penultimate_relative_pcp) + '\n'
                tmp_log_line += 'melody penultimate rpcp: ' + str(p.melody_chord_segments[-2].relative_pcp) + '\n'
                tmp_log_line += 'melody important penultimate rpcp: ' + str(p.melody_chord_segments[-2].important_relative_pcp) + '\n'
                tmp_log_line += 'Penultimate chord score: ' + "{:.4f}".format(penultimate_score)
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # cadence_score = (0.8*final_score + 0.2*penultimate_score)*c.probability
    cadence_score = (0.5*final_score + 0.5*penultimate_score)
    # include the cadence probability factor
    cadence_score = cadence_score*pow(c.probability, 0.01)
    # cadence_score = np.power(final_score, 1)*np.power(penultimate_score, 3)*c.probability
    if np.isnan(cadence_score):
        cadence_score = 0.0
    # log eval entire cadence
    if logging:
        # log final score
        tmp_log_line = 'OVERALL score: ' + "{:.4f}".format(cadence_score) + '\n'
        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    return cadence_score
# end evaluate_cadence_in_phrase
def get_best_matching_cadence(p, mode, m, logging=False):
    ' gets phrase structure from melody p and mode from learned idiom m '
    ' and returns the best matching cadence object '
    ' m is only necessary for getting the filename for logging '
    c_out = []
    overall_score = -1
    # check if phrase has no chords
    if len(p.melody_chord_segments) < 1:
        print('WEIRD melody input in CM_similarity_functions.py: phrase has no chord segments')
        # log selected mode
        if logging:
            tmp_log_line = 'WEIRD melody input in CM_GN_cadence_functions.py: phrase has no chord segments'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    else: # if it does have at least one chord
        # check if phrase is final or intermediate
        all_cads = mode.cadences['intermediate']
        other_cads = mode.cadences['final']
        # check if it's a 'final' cadence or if list of intermediate cadences is empty
        if p.level > 1 or len( all_cads.cadences_dictionary.values() ) == 0:
            all_cads = mode.cadences['final']
            other_cads = mode.cadences['intermediate']
            # log cadence info
            if logging:
                tmp_log_line = 'Selecting final cadences initially'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        else:
            # log cadence info
            if logging:
                tmp_log_line = 'Selecting intermediate cadences initially'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # run through all cadences and evaluate them
        cad_keys = list( all_cads.cadences_dictionary.keys() )
        for i in range( len( cad_keys ) ):
            c = all_cads.cadences_dictionary[ cad_keys[i] ]
            # log which cadence is evaluated
            if logging:
                # log constraint score
                tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
            tmp_score = evaluate_cadence_in_phrase( p, c, m, logging=logging )
            if overall_score < tmp_score:
                overall_score = tmp_score
                c_out = c
            # log best cadence so far
            if logging:
                # log constraint score
                tmp_log_line = 'Best cadence: ' + c_out.label + '\n'
                tmp_log_line += 'Best score: ' + "{:.4f}".format(overall_score) + '\n'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # check if score too low and use alternative cadences
        if overall_score < 0.7:
            # log going for alternatives - score too low
            if logging:
                # log constraint score
                tmp_log_line = 'Score too low - aiming for alternative set of cadences (intermediate/final)' + '\n'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
            # run throuth all alternative cadences and evaluate them
            cad_keys = list( other_cads.cadences_dictionary.keys() )
            for i in range( len( cad_keys ) ):
                c = other_cads.cadences_dictionary[ cad_keys[i] ]
                # log which cadence is evaluated
                if logging:
                    # log constraint score
                    tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
                    prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
                tmp_score = evaluate_cadence_in_phrase( p, c, m, logging=logging )
                if overall_score < tmp_score:
                    overall_score = tmp_score
                    c_out = c
                # log best cadence so far
                if logging:
                    # log constraint score
                    tmp_log_line = 'Best cadence: ' + c_out.label + '\n'
                    tmp_log_line += 'Best score: ' + "{:.4f}".format(overall_score) + '\n'
                    prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    return c_out
# end get_best_matching_cadence