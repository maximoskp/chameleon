#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:59:35 2018

@author: maximoskaliakatsos-papakostas
"""

import CM_similarity_functions as smf
import numpy as np
import CM_Misc_Aux_functions as maf

def get_cadence_with_final_constraint(cs_penultimate, cs_final, all_cads, other_cads):
    ' cs is a the entire chord segment, for getting not only '
    ' the chord pcp, but also the melody notes '
    best_scor = 0
    best_idx = 0
    best_cads = all_cads
    # run throuth all cadences and evaluate them
    for i in range( len( all_cads.all_cadence_structures ) ):
        c = all_cads.all_cadence_structures[i]
        # get score based on final matching with constraint
        print('c.pen: ', c.penultimate_relative_pcp)
        constraint_score = (np.corrcoef(c.final_relative_pcp, cs_final.gct_rpcp)[0,1] + 1)/2
        melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_penultimate.relative_pcp, cs_penultimate.important_relative_pcp, c.penultimate_relative_pcp )
        # # __DEBUG__
        # print(' ==================== ')
        # print('c.pen: ', c.penultimate_relative_pcp)
        # print('cs.re: ', cs.relative_pcp)
        # print('melody_score: ', melody_score)
        # # __DEBUG__
        tmp_score = 0.2*constraint_score + 0.8*melody_score
        if best_scor < tmp_score:
            best_scor = tmp_score
            best_idx = i
    # check if score too low and use alternative cadences
    # __DEBUG__
    print('best_scor: ', best_scor)
    # __DEBUG__
    if best_scor < 0.7:
        best_cads = other_cads
        # run throuth all alternative cadences and evaluate them
        for i in range( len( other_cads.all_cadence_structures ) ):
            c = other_cads.all_cadence_structures[i]
            # get score based on final matching with constraint
            constraint_score = (np.corrcoef(c.final_relative_pcp, cs_final.gct_rpcp)[0,1] + 1)/2
            melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_penultimate.relative_pcp, cs_penultimate.important_relative_pcp, c.penultimate_relative_pcp )
            tmp_score = 0.2*constraint_score + 0.8*melody_score
            if best_scor < tmp_score:
                best_scor = tmp_score
                best_idx = i
    return best_cads.all_cadence_structures[best_idx]
# end get_cadence_with_final_constraint
def get_cadence_with_penultimate_constraint(cs_penultimate, cs_final, all_cads, other_cads):
    ' cs is a the entire chord segment, for getting not only '
    ' the chord pcp, but also the melody notes '
    best_scor = 0
    best_idx = 0
    best_cads = all_cads
    # run throuth all cadences and evaluate them
    for i in range( len( all_cads.all_cadence_structures ) ):
        c = all_cads.all_cadence_structures[i]
        # get score based on final matching with constraint
        constraint_score = (np.corrcoef(c.penultimate_relative_pcp, cs_penultimate.gct_rpcp)[0,1] + 1)/2
        melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_final.relative_pcp, cs_final.important_relative_pcp, c.final_relative_pcp )
        tmp_score = 0.2*constraint_score + 0.8*melody_score
        if best_scor < tmp_score:
            best_scor = tmp_score
            best_idx = i
    # check if score too low and use alternative cadences
    if best_scor < 0.7:
        best_cads = other_cads
        # run throuth all alternative cadences and evaluate them
        for i in range( len( other_cads.all_cadence_structures ) ):
            c = other_cads.all_cadence_structures[i]
            # get score based on final matching with constraint
            constraint_score = (np.corrcoef(c.final_relative_pcp, cs_penultimate.gct_rpcp)[0,1] + 1)/2
            melody_score = smf.evaluate_melody_chord_rpcp_matching( cs_final.relative_pcp, cs_final.important_relative_pcp, c.penultimate_relative_pcp )
            tmp_score = 0.2*constraint_score + 0.8*melody_score
            if best_scor < tmp_score:
                best_scor = tmp_score
                best_idx = i
    return all_cads.all_cadence_structures[best_idx]
# end get_cadence_with_penultimate_constraint

def apply_cadences_to_melody_from_idiom(m,idiom):
    print('applying cadences')
    # run through all phrases
    for p in m.phrases:
        # get proper mode that corresponds to the mode of the phrase
        mode = smf.get_best_matching_mode( p.tonality.mode_pcp, idiom )
        # check if any of the last two chords in constrained
        constrained_cadence = False
        if len(p.melody_chord_segments) == 0:
            print('WEIRD melody input in CM_GN_cadence_functions.py: phrase has no chord segments')
        elif len(p.melody_chord_segments) == 1 and p.melody_chord_segments[-1].is_constraint:
            # if only one chord has been included and it is constrained, just use this chord
            print('no need to add cadence, there is only one chord and it is constrained!')
            constrained_cadence = True
        else:
            # check if we have more than two chords
            if len(p.melody_chord_segments) >= 2:
                # check if both ending chords are constrained
                if p.melody_chord_segments[-1].is_constraint and p.melody_chord_segments[-2].is_constraint:
                    # leave them as they are
                    print('no need to add cadence, both ending chords are constrained!')
                    constrained_cadence = True
                # check if any of two ending chords is constrained
                if p.melody_chord_segments[-1].is_constraint or p.melody_chord_segments[-2].is_constraint:
                    # check if final chord is constrained
                    if p.melody_chord_segments[-1].is_constraint:
                        # get cadence that best matches the final chord
                        constrained_cadence = True
                        print('final chord of cadence is constrained')
                        # check if phrase is final or intermediate
                        all_cads = mode.cadences['intermediate']
                        other_cads = mode.cadences['final']
                        # check if it's a 'final' cadence or if list of intermediate cadences is empty
                        if p.level > 1 or len( all_cads.cadences_dictionary.values() ) == 0:
                            all_cads = mode.cadences['final']
                            other_cads = mode.cadences['intermediate']
                        best_cadence = get_cadence_with_final_constraint( p.melody_chord_segments[-2], p.melody_chord_segments[-1], all_cads, other_cads )
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
                        # check if phrase is final or intermediate
                        all_cads = mode.cadences['intermediate']
                        other_cads = mode.cadences['final']
                        # check if it's a 'final' cadence or if list of intermediate cadences is empty
                        if p.level > 1 or len( all_cads.cadences_dictionary.values() ) == 0:
                            all_cads = mode.cadences['final']
                            other_cads = mode.cadences['intermediate']
                        best_cadence = get_cadence_with_penultimate_constraint( p.melody_chord_segments[-2], p.melody_chord_segments[-1], all_cads, other_cads )
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
            best_cadence = smf.get_best_matching_cadence( p, mode )
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