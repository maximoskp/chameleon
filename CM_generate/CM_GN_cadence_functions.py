#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:59:35 2018

@author: maximoskaliakatsos-papakostas
"""

import CM_similarity_functions as smf
import numpy as np
import CM_Misc_Aux_functions as maf
import os
cwd = os.getcwd()
import sys
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

def get_cadence_with_final_constraint(cs_penultimate, cs_final, all_cads, other_cads, m, logging=False):
    ' cs is a the entire chord segment, for getting not only '
    ' the chord pcp, but also the melody notes '
    ' m is only necessary for getting the filename for logging '
    best_scor = 0
    best_idx = 0
    best_cads = all_cads
    # run throuth all cadences and evaluate them
    cad_keys = list( all_cads.cadences_dictionary.keys() )
    for i in range( len( cad_keys ) ):
        c = all_cads.cadences_dictionary[ cad_keys[i] ]
        # get score based on final cadence chord matching with constraint
        constraint_score = (np.corrcoef(c.final_relative_pcp, cs_final.gct_rpcp)[0,1] + 1)/2
        # log constraint_score
        if logging:
            # log constraint score
            tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
            tmp_log_line += 'Constraint final chord: ' + cs_final.gct_label + '\n'
            tmp_log_line += 'Cadence final chord: ' + c.final_gct + '\n'
            tmp_log_line += 'Constraint score: ' + "{:.4f}".format(constraint_score)
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_penultimate.relative_pcp, cs_penultimate.important_relative_pcp, c.penultimate_relative_pcp )
        tmp_score = 0.2*constraint_score + 0.8*melody_score
        # include the cadence probability factor
        tmp_score = tmp_score*pow(c.probability, 0.01)
        if best_scor < tmp_score:
            best_scor = tmp_score
            best_idx = i
        # log eval all cadences
        if logging:
            # log penultimate score
            tmp_log_line = 'Cadence penultimate chord: ' + c.penultimate_gct + '\n'
            tmp_log_line += 'cadence penultimate rpcp: ' + str(c.penultimate_relative_pcp) + '\n'
            tmp_log_line += 'melody penultimate rpcp: ' + str(cs_penultimate.relative_pcp) + '\n'
            tmp_log_line += 'melody important penultimate rpcp: ' + str(cs_penultimate.important_relative_pcp) + '\n'
            tmp_log_line += 'Penultimate score: ' + "{:.4f}".format(melody_score) + '\n'
            tmp_log_line += 'OVERALL score: ' + "{:.4f}".format(tmp_score) + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # log best cadence info
    if logging:
        # log best score
        tmp_log_line += 'Best score: (idx: ' + str(best_idx) + ' ): ' + "{:.4f}".format(best_scor) + '\n'
        tmp_log_line += 'Best cadence: (idx: ' + str(best_idx) + ' ): ' + best_cads.cadences_dictionary[ cad_keys[best_idx] ].label + '\n'
        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # check if score too low and use alternative cadences
    if best_scor < 0.7:
        # log going for alternatives - score too low
        if logging:
            # log constraint score
            tmp_log_line = 'Score too low - aiming for alternative set of cadences (intermediate/final)' + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        best_cads = other_cads
        # run throuth all alternative cadences and evaluate them
        cad_keys = list( other_cads.cadences_dictionary.keys() )
        for i in range( len( cad_keys ) ):
            c = other_cads.cadences_dictionary[ cad_keys[i] ]
            # get score based on final cadence chord matching with constraint
            constraint_score = (np.corrcoef(c.final_relative_pcp, cs_final.gct_rpcp)[0,1] + 1)/2
            # log constraint_score
            if logging:
                # log constraint score
                tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
                tmp_log_line += 'Constraint final chord: ' + cs_final.gct_label + '\n'
                tmp_log_line += 'Cadence final chord: ' + c.final_gct + '\n'
                tmp_log_line += 'score: ' + "{:.4f}".format(constraint_score)
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
            melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_penultimate.relative_pcp, cs_penultimate.important_relative_pcp, c.penultimate_relative_pcp )
            tmp_score = 0.2*constraint_score + 0.8*melody_score
            # include the cadence probability factor
            tmp_score = tmp_score*pow(c.probability, 0.01)
            if best_scor < tmp_score:
                best_scor = tmp_score
                best_idx = i
            # log eval all cadences
            if logging:
                # log penultimate score
                tmp_log_line = 'Cadence penultimate chord: ' + c.penultimate_gct + '\n'
                tmp_log_line += 'cadence penultimate rpcp: ' + str(c.penultimate_relative_pcp) + '\n'
                tmp_log_line += 'melody penultimate rpcp: ' + str(cs_penultimate.relative_pcp) + '\n'
                tmp_log_line += 'melody important penultimate rpcp: ' + str(cs_penultimate.important_relative_pcp) + '\n'
                tmp_log_line += 'Penultimate score: ' + "{:.4f}".format(melody_score) + '\n'
                tmp_log_line += 'OVERALL score: ' + "{:.4f}".format(tmp_score) + '\n'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # log best cadence info
        if logging:
            # log best score
            tmp_log_line += 'Best score: (idx: ' + str(best_idx) + ' ): ' + "{:.4f}".format(best_scor) + '\n'
            tmp_log_line += 'Best cadence: (idx: ' + str(best_idx) + ' ): ' + best_cads.cadences_dictionary[ cad_keys[best_idx] ].label + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    return best_cads.cadences_dictionary[ cad_keys[best_idx] ]
# end get_cadence_with_final_constraint
def get_cadence_with_penultimate_constraint(cs_penultimate, cs_final, all_cads, other_cads, m, logging=False):
    ' cs is a the entire chord segment, for getting not only '
    ' the chord pcp, but also the melody notes '
    ' m is only necessary for getting the filename for logging '
    best_scor = 0
    best_idx = 0
    best_cads = all_cads
    # run throuth all cadences and evaluate them
    cad_keys = list( all_cads.cadences_dictionary.keys() )
    for i in range( len( cad_keys ) ):
        c = all_cads.cadences_dictionary[ cad_keys[i] ]
        # get score based on final matching with constraint
        constraint_score = (np.corrcoef(c.penultimate_relative_pcp, cs_penultimate.gct_rpcp)[0,1] + 1)/2
        # log constraint_score
        if logging:
            # log constraint score
            tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
            tmp_log_line += 'Constraint penultimate chord: ' + cs_penultimate.gct_label + '\n'
            tmp_log_line += 'Cadence penultimate chord: ' + c.penultimate_gct + '\n'
            tmp_log_line += 'Constraint score: ' + "{:.4f}".format(constraint_score)
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_final.relative_pcp, cs_final.important_relative_pcp, c.final_relative_pcp )
        tmp_score = 0.2*constraint_score + 0.8*melody_score
        # include the cadence probability factor
        tmp_score = tmp_score*pow(c.probability, 0.01)
        if best_scor < tmp_score:
            best_scor = tmp_score
            best_idx = i
        # log eval all cadences
        if logging:
            # log final score
            tmp_log_line = 'Cadence final chord: ' + c.final_gct + '\n'
            tmp_log_line += 'cadence final rpcp: ' + str(c.final_relative_pcp) + '\n'
            tmp_log_line += 'melody final rpcp: ' + str(cs_final.relative_pcp) + '\n'
            tmp_log_line += 'melody important final rpcp: ' + str(cs_final.important_relative_pcp) + '\n'
            tmp_log_line += 'Final score: ' + "{:.4f}".format(melody_score) + '\n'
            tmp_log_line += 'OVERALL score: ' + "{:.4f}".format(tmp_score) + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # log best cadence info
    if logging:
        # log best score
        tmp_log_line += 'Best score: (idx: ' + str(best_idx) + ' ): ' + "{:.4f}".format(best_scor) + '\n'
        tmp_log_line += 'Best cadence: (idx: ' + str(best_idx) + ' ): ' + best_cads.cadences_dictionary[ cad_keys[best_idx] ].label + '\n'
        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # check if score too low and use alternative cadences
    if best_scor < 0.7:
        # log going for alternatives - score too low
        if logging:
            # log constraint score
            tmp_log_line = 'Score too low - aiming for alternative set of cadences (intermediate/final)' + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        best_cads = other_cads
        # run throuth all alternative cadences and evaluate them
        cad_keys = list( other_cads.cadences_dictionary.keys() )
        for i in range( len( cad_keys ) ):
            c = other_cads.cadences_dictionary[ cad_keys[i] ]
            # get score based on final matching with constraint
            constraint_score = (np.corrcoef(c.final_relative_pcp, cs_penultimate.gct_rpcp)[0,1] + 1)/2
            # log constraint_score
            if logging:
                # log constraint score
                tmp_log_line = str(i) + ' -- evaluating cadence: ' + c.label + '\n'
                tmp_log_line += 'Constraint penultimate chord: ' + cs_penultimate.gct_label + '\n'
                tmp_log_line += 'Cadence penultimate chord: ' + c.penultimate_gct + '\n'
                tmp_log_line += 'score: ' + "{:.4f}".format(constraint_score)
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
            melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_final.relative_pcp, cs_final.important_relative_pcp, c.penultimate_relative_pcp )
            tmp_score = 0.2*constraint_score + 0.8*melody_score
            # include the cadence probability factor
            tmp_score = tmp_score*pow(c.probability, 0.01)
            if best_scor < tmp_score:
                best_scor = tmp_score
                best_idx = i
            # log eval all cadences
            if logging:
                # log penultimate score
                tmp_log_line = 'Cadence final chord: ' + c.final_gct + '\n'
                tmp_log_line += 'cadence final rpcp: ' + str(c.final_relative_pcp) + '\n'
                tmp_log_line += 'melody final rpcp: ' + str(cs_final.relative_pcp) + '\n'
                tmp_log_line += 'melody important final rpcp: ' + str(cs_final.important_relative_pcp) + '\n'
                tmp_log_line += 'Final score: ' + "{:.4f}".format(melody_score) + '\n'
                tmp_log_line += 'OVERALL score: ' + "{:.4f}".format(tmp_score) + '\n'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # log best cadence info
        if logging:
            # log best score
            tmp_log_line += 'Best score: (idx: ' + str(best_idx) + ' ): ' + "{:.4f}".format(best_scor) + '\n'
            tmp_log_line += 'Best cadence: (idx: ' + str(best_idx) + ' ): ' + best_cads.cadences_dictionary[ cad_keys[best_idx] ].label + '\n'
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    return best_cads.cadences_dictionary[ cad_keys[best_idx] ]
# end get_cadence_with_penultimate_constraint

def apply_cadences_to_melody_from_idiom(m,idiom, mode_in='Auto',logging=False):
    print('applying cadences')

    # log that it's about cadences
    if logging:
        tmp_log_line = 'CADENCES =================================== ' + '\n'
        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
    # run through all phrases
    for p in m.phrases:
        # check what mode option is given by the user
        # MODE SELECTION SNIPPET ==================================================
        if mode_in == 'Auto':
            # get proper mode that corresponds to the mode of the phrase
            mode = smf.get_best_matching_mode( p.tonality.mode_pcp, idiom )
        else:
            # get mode names
            mode_keys = list(idiom.modes.keys())
            if mode_in not in mode_keys:
                # if not in existing modes, select as if auto was given
                mode = smf.get_best_matching_mode( p.tonality.mode_pcp, idiom )
            else:
                mode = idiom.modes[ mode_in ]
        # MODE SELECTION SNIPPET ==================================================
        # log selected mode
        if logging:
            tmp_log_line = 'NEW PHRASE ========================= ' + '\n'
            tmp_log_line += 'User-given phrase mode: ' + str(p.tonality.mode_pcp) + '\n'
            tmp_log_line += 'CADENCES selected mode: ' + mode.mode_name
            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        # check if any of the last two chords in constrained
        constrained_cadence = False
        if len(p.melody_chord_segments) == 0:
            print('WEIRD melody input in CM_GN_cadence_functions.py: phrase has no chord segments')
            # log selected mode
            if logging:
                tmp_log_line = 'WEIRD melody input in CM_GN_cadence_functions.py: phrase has no chord segments'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
        elif len(p.melody_chord_segments) == 1 and p.melody_chord_segments[-1].is_constraint:
            # if only one chord has been included and it is constrained, just use this chord
            print('no need to add cadence, there is only one chord and it is constrained!')
            # log selected mode
            if logging:
                tmp_log_line = 'No need to add cadence, there is only one chord and it is constrained!'
                prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
            constrained_cadence = True
        else:
            # check if we have more than two chords
            if len(p.melody_chord_segments) >= 2:
                # check if both ending chords are constrained
                if p.melody_chord_segments[-1].is_constraint and p.melody_chord_segments[-2].is_constraint:
                    # leave them as they are
                    print('no need to add cadence, both ending chords are constrained!')
                    constrained_cadence = True
                    # log cadence info
                    if logging:
                        tmp_log_line = 'No need to add cadence, both ending chords are constrained!'
                        prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
                # check if any of two ending chords is constrained
                if p.melody_chord_segments[-1].is_constraint or p.melody_chord_segments[-2].is_constraint:
                    # check if final chord is constrained
                    if p.melody_chord_segments[-1].is_constraint:
                        # get cadence that best matches the final chord
                        constrained_cadence = True
                        print('final chord of cadence is constrained')
                        # log cadence info
                        if logging:
                            tmp_log_line = 'Final chord of cadence is constrained'
                            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
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
                        best_cadence = get_cadence_with_final_constraint( p.melody_chord_segments[-2], p.melody_chord_segments[-1], all_cads, other_cads, m, logging=logging )
                        # p.melody_chord_segments[-1].gct_chord = best_cadence.gcts_pair[-1]
                        # p.melody_chord_segments[-1].gct_label = np2str( best_cadence.gcts_pair[-1] )
                        # p.melody_chord_segments[-1].gct_rpcp = gct2relpcp( best_cadence.gcts_pair[-1] )
                        # p.melody_chord_segments[-1].is_constraint = True
                        p.melody_chord_segments[-2].gct_chord = best_cadence.gcts_pair[-2]
                        p.melody_chord_segments[-2].gct_label = maf.np2str( best_cadence.gcts_pair[-2] )
                        p.melody_chord_segments[-2].gct_rpcp = maf.gct2relpcp( best_cadence.gcts_pair[-2] )
                        p.melody_chord_segments[-2].is_constraint = True
                    else:
                        # get cadence that best matches the penultimate chord
                        constrained_cadence = True
                        print('penultimate chord of cadence is constrained')
                        # log cadence info
                        if logging:
                            tmp_log_line = 'Penultimate chord of cadence is constrained'
                            prt.print_log_line( m.harmonisation_file_name, tmp_log_line )
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
                        best_cadence = get_cadence_with_penultimate_constraint( p.melody_chord_segments[-2], p.melody_chord_segments[-1], all_cads, other_cads, m, logging=logging )
                        p.melody_chord_segments[-1].gct_chord = best_cadence.gcts_pair[-1]
                        p.melody_chord_segments[-1].gct_label = maf.np2str( best_cadence.gcts_pair[-1] )
                        p.melody_chord_segments[-1].gct_rpcp = maf.gct2relpcp( best_cadence.gcts_pair[-1] )
                        p.melody_chord_segments[-1].is_constraint = True
                        # p.melody_chord_segments[-2].gct_chord = best_cadence.gcts_pair[-2]
                        # p.melody_chord_segments[-2].gct_label = np2str( best_cadence.gcts_pair[-2] )
                        # p.melody_chord_segments[-2].gct_rpcp = gct2relpcp( best_cadence.gcts_pair[-2] )
                        # p.melody_chord_segments[-2].is_constraint = True
        # if not constrained cadence, find the best cadence based on melody notes
        if not constrained_cadence:
            # get best matching cadence in mode
            best_cadence = smf.get_best_matching_cadence( p, mode, m, logging=logging )
            # apply best cadence to phrase
            if len(p.melody_chord_segments) == 0:
                print('WEIRD melody input in CM_GN_cadence_functions.py: phrase has no chord segments')
            elif len(p.melody_chord_segments) == 1:
                p.melody_chord_segments[-1].gct_chord = best_cadence.gcts_pair[-1]
                p.melody_chord_segments[-1].gct_label = maf.np2str( best_cadence.gcts_pair[-1] )
                p.melody_chord_segments[-1].gct_rpcp = maf.gct2relpcp( best_cadence.gcts_pair[-1] )
                p.melody_chord_segments[-1].is_constraint = True
            else:
                p.melody_chord_segments[-1].gct_chord = best_cadence.gcts_pair[-1]
                p.melody_chord_segments[-1].gct_label = maf.np2str( best_cadence.gcts_pair[-1] )
                p.melody_chord_segments[-1].gct_rpcp = maf.gct2relpcp( best_cadence.gcts_pair[-1] )
                p.melody_chord_segments[-1].is_constraint = True
                p.melody_chord_segments[-2].gct_chord = best_cadence.gcts_pair[-2]
                p.melody_chord_segments[-2].gct_label = maf.np2str( best_cadence.gcts_pair[-2] )
                p.melody_chord_segments[-2].gct_rpcp = maf.gct2relpcp( best_cadence.gcts_pair[-2] )
                p.melody_chord_segments[-2].is_constraint = True
    return m
# end apply_cadences_to_melody_from_idiom