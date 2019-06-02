#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 07:50:10 2018

@author: maximoskaliakatsos-papakostas
"""

import numpy as np
import music21 as m21
import copy
# import NN_VL_model as nnvl

def make_harmonisation_stream(m, idiom):
    output_stream = m21.stream.Stream()
    melody_part = m.input_stream.parts[0]
    harmonic_rhythm_part = copy.deepcopy( m.input_stream.parts[2] )
    # concatenate all chord segments from all phrases
    chord_segments = []
    for p in m.phrases:
        chord_segments.extend( p.melody_chord_segments )
    # __NN_VL__
    '''
    matrix, melody_bin, harmony_options_bin, gct_completion, model = apply_NN_voice_leading(chord_segments)
    # keep a chord_segments copy to return
    s = copy.deepcopy( chord_segments )
    '''
    # __NN_VL__
    # counter for chord segment
    tmp_count = 0
    for measure in harmonic_rhythm_part.getElementsByClass('Measure'):
        for note in measure.getElementsByClass('Note'):
            chord_segments[tmp_count].chord.offset = note.offset
            chord_segments[tmp_count].chord.duration = note.duration
            note.activeSite.replace(note, chord_segments[tmp_count].chord)
            tmp_count += 1
    output_stream.insert( 0, melody_part )
    output_stream.insert( 0, harmonic_rhythm_part )
    m.output_stream = output_stream
    return m
# end make_harmonisation_stream
def apply_simple_voice_leading(m, idiom):
    # get lowest midi pitch of melody
    lowest_melody_pitch = 127
    for n in m.melodyNotes:
        if n.pitch.midi < lowest_melody_pitch:
            lowest_melody_pitch = n.pitch.midi
    # adjust lowet octave limit
    lowest_octave_limit = 120
    while lowest_octave_limit >= lowest_melody_pitch:
        lowest_octave_limit  = lowest_octave_limit - 12
    for p in m.phrases:
        # get tonality key
        key = p.tonality.key
        # for each chord segment get gct chord
        for cs in p.melody_chord_segments:
            g = cs.gct_chord
            chord_root = g[0] + key
            chord = np.array( g[1:] ) + chord_root + lowest_octave_limit - 12
            #  make bass one octave lower
            chord[0] = chord[0] - 12
            # keep at most 4 notes
            if len( chord ) > 4:
                chord = chord[:4]
            chord_final = chord.astype(int).tolist()
            cs.chord = m21.chord.Chord( chord_final )
    m = make_harmonisation_stream(m, idiom)
    return m

# bidirectional bvl functions ============================================
# we also need it outside the class
def make_neutral_bbvl(c):
    # return a bbvl structure with uniform probabilities for all possibilities
    stats = {}
    stats['inversions'] = np.ones( len(c)-1 )/(len(c)-1)
    num_octaves = 6
    stats['mel2bass'] = np.ones(12*num_octaves)/(12*num_octaves)
    stats['from_bvl'] = np.ones( ( 25 , len(c)-1 ) )/(25*(len(c)-1 ))
    stats['to_bvl'] = np.ones( ( 25 , len(c)-1 ) )/(25*(len(c)-1 ))
    return stats
# end make_neutral_bbvl
class BBVL:
    ''' bidirectional bass voice leading class '''
    def __init__(self, m, idiom, use_GCT_grouping, mode_in='Auto'):
        # keep the entire melodic and idioms structures just to return them
        self.m = m
        self.idiom = m
        # for each phrase, get the gcts and the respective bbvl objects
        self.gct_nps = []
        self.gct_pcs = []
        self.gct_labels = []
        self.bbvl_objects = []
        self.melody_midis = []
        for p in m.phrases:
            # get mode of phrase
            phrase_mode = idiom.modes[ p.idiom_mode_label ]
            for cs in p.melody_chord_segments:
                self.gct_nps.append( cs.gct_chord )
                self.gct_pcs.append( np.mod( cs.gct_chord[0]+p.tonality.key+cs.gct_chord[1:] , 12 ) )
                self.gct_labels.append( cs.gct_label )
                # get the proper bbvl dictionary, depending on if gct grouping is used
                if use_GCT_grouping:
                    bbvl_dict = phrase_mode.gct_group_info.gct_vl_dict
                else:
                    bbvl_dict = phrase_mode.gct_info.gct_vl_dict
                # there is possibility that the user has given a constraint that is not included in gct labels
                if cs.gct_label in list( bbvl_dict.keys() ):
                    self.bbvl_objects.append( bbvl_dict[ cs.gct_label ] )
                else:
                    # get a neutral bbvl object that corresponds to the size of this gct
                    self.bbvl_objects.append( self.make_neutral_bbvl( cs.gct_chord ) )
                self.melody_midis.append( cs.melody_midi )
        # now that we have all the harmonisation information in place, make the aux matrices
        # make the matrix with the melodic height constraints
        self.bheight = self.melodic_height_constraints(self.melody_midis)
    # end constructor
    def make_neutral_bbvl(self, c):
        # return a bbvl structure with uniform probabilities for all possibilities
        stats = {}
        stats['inversions'] = np.ones( len(c)-1 )/(len(c)-1)
        num_octaves = 6
        stats['mel2bass'] = np.ones(12*num_octaves)/(12*num_octaves)
        stats['from_bvl'] = np.ones( ( 25 , len(c)-1 ) )/(25*(len(c)-1 ))
        stats['to_bvl'] = np.ones( ( 25 , len(c)-1 ) )/(25*(len(c)-1 ))
        return stats
    # end make_neutral_bbvl
    def melodic_height_constraints(self, m):
        mhc = np.ones( (128, len(m)) )
        laska = 12
        for i in range( len( m ) ):
            mhc[ (np.min(m[i])-laska): , i ] = 0
        return mhc
    # end melodic_height_constraints
    def bbvl_viterbi(self):
        # weights of VL importance
        # since we're dealing with probabilities (<1),
        # weights <1 indicate importance - >1 umimportance
        # a) inversion
        a_inv_w = 0.95
        # b) melodic height probabilites
        b_mhi_w = 0.9
        # c) from bvl of current
        c_fbvl_w = 0.99
        # d) to bvl of previous
        d_tbvl_w = 0.99
        # e) delta of previous
        e_prdelt_w = 1.0
        delta = np.zeros( (128, len(self.gct_nps)) )
        psi = np.zeros( (128, len(self.gct_nps)) )
        pathIDXs = np.zeros( len(self.gct_nps) )
        t = 0
        # get melody-related probabilities
        # get melody lowest limit
        mll = np.min(self.melody_midis[t])
        # get the mel2bass distribution
        m2b = self.bbvl_objects[t]['mel2bass']
        # reverse it so that we get distance from the melody and not from zeros
        m2b = m2b[::-1]
        # adjust m2b to melody
        m2b_probs = np.zeros(128)
        m2b_probs[max([0,mll-m2b.shape[0]]):mll] = m2b[max([0,m2b.shape[0]-mll]):]
        # initial probability for each midi note
        for i in range(128):
            # check if pc in gct
            if i%12 in self.gct_pcs[t]:
                # get idx of gct element
                idx = np.where( i%12==self.gct_pcs[t] )[0][0]
                # get the inversion probability
                a_inv_prob = self.bbvl_objects[t]['inversions'][idx]
                # get the m2b prob for this pitch
                b_mel_prob = m2b_probs[i]
                # assign weights before computing delta
                a_inv_prob = np.power(a_inv_prob, a_inv_w)
                b_mel_prob = np.power(b_mel_prob, b_mhi_w)
                # compute delta value
                delta[i,t] = a_inv_prob*b_mel_prob
        # arbitrary psi value since there is no predecessor to t=0
        psi[:,t] = 0

        # start building trellis
        # we will need to combine probabilities for:
        # a) inversion
        # b) melodic height probabilites
        # c) from bvl of current
        # d) to bvl of previous
        # e) delta of previous
        for t in range(1,len(self.gct_nps),1):
            # get melody-related probabilities
            # get melody lowest limit
            mll = np.min(self.melody_midis[t])
            # get the mel2bass distribution
            m2b = self.bbvl_objects[t]['mel2bass']
            # reverse it so that we get distance from the melody and not from zeros
            m2b = m2b[::-1]
            # adjust m2b to melody
            m2b_probs = np.zeros(128)
            m2b_probs[max([0,mll-m2b.shape[0]]):mll] = m2b[max([0,m2b.shape[0]-mll]):]
            # get previous gct_pcs
            prev_gct_pcs = self.gct_pcs[t-1]
            # for all possible CURRENT pitches
            for i in range(128):
                # check if pitch is in CURRENT gct
                if i%12 in self.gct_pcs[t]:
                    # get idx of gct element
                    idx = np.where( i%12==self.gct_pcs[t] )[0][0]
                    # get the inversion probability
                    a_inv_prob = self.bbvl_objects[t]['inversions'][idx]
                    # get the m2b prob for this pitch
                    b_mel_prob = m2b_probs[i]
                    # get from bvl for the current
                    from_bvl = self.bbvl_objects[t]['from_bvl'][:,idx]
                    # get all the possible delta for getting to current pitch
                    tmp_delta = np.zeros(128)
                    # for all possible PREVIOUS pitches
                    for j in range(128):
                        if j%12 in prev_gct_pcs:
                            # get index of previous gct
                            prev_idx = np.where( j%12==prev_gct_pcs )[0][0]
                            # get to_bvl probabilities
                            to_bvl = self.bbvl_objects[t-1]['to_bvl'][:,prev_idx]
                            # e) delta of previous
                            # get the delta value
                            e_delta = delta[j,t-1]
                            # compute pitch differences and get values from to/from_bvl arrays
                            pitch_diff_idx = (int)(i-j + (len(from_bvl)-1)/2)
                            if pitch_diff_idx < 0 :
                                pitch_diff_idx = 0
                            if pitch_diff_idx >= len(from_bvl):
                                pitch_diff_idx = len(from_bvl) - 1
                            # get to/from_bvl probs through pitch_diff_idx
                            c_to_bvl_prob = to_bvl[pitch_diff_idx]
                            d_from_bvl_prob = from_bvl[pitch_diff_idx]
                            # assign weights before computing delta
                            a_inv_prob = np.power(a_inv_prob, a_inv_w)
                            b_mel_prob = np.power(b_mel_prob, b_mhi_w)
                            c_to_bvl_prob = np.power(c_to_bvl_prob, c_fbvl_w)
                            d_from_bvl_prob = np.power(d_from_bvl_prob, d_tbvl_w)
                            e_delta = np.power(e_delta, e_prdelt_w)
                            # find maximum before assigning to delta
                            tmp_delta[j] = a_inv_prob*b_mel_prob*c_to_bvl_prob*d_from_bvl_prob*e_delta
                    # compute delta and psi value based on maximum value
                    delta[i,t] = np.max( tmp_delta )
                    psi[i,t] = np.argmax( tmp_delta )
            # normalise delta column
            if np.sum(delta[:,t]) != 0:
                delta[:,t] = delta[:,t]/np.sum(delta[:,t])
        # back-track trellis
        pathIDXs[ len(self.gct_nps)-1 ] = int( np.argmax( delta[:,len(self.gct_nps)-1] ) )
        for t in range( len(self.gct_nps)-2, -1, -1 ):
            pathIDXs[t] = (int)( psi[ int(pathIDXs[t+1]) , t+1 ] )
            pathIDXs = pathIDXs.astype(int)
        print('vl pathIDXs: ', pathIDXs)
        return pathIDXs.tolist()
    # end bbvl_viterbi
    def apply_bbvl(self):
        b = self.bbvl_viterbi()
        # assign bass to harmonisation - TODO: fill intermediate voices
        tmp_idx = 0
        for p in self.m.phrases:
            for cs in p.melody_chord_segments:
                # temporary list of gct pcs for crossing out what has been used
                tmp_gct_pcs = copy.deepcopy( self.gct_pcs[ tmp_idx ] )
                # temporary array with midi pitches
                tmp_midis = [ b[ tmp_idx ] ]
                # remove bass note
                to_delete = np.where( tmp_gct_pcs==(b[ tmp_idx ]%12) )[0][0]
                tmp_gct_pcs = np.delete(tmp_gct_pcs, to_delete)
                # keep at most 4 notes - or 3 given that the bass has already been considered
                if len( tmp_gct_pcs ) > 3:
                    tmp_gct_pcs = tmp_gct_pcs[:3]
                # pass all other notes
                for pc in tmp_gct_pcs:
                    # get melody lowest limit
                    mll = np.min(self.melody_midis[tmp_idx])
                    # get base-height according to melody octave
                    mel_oct = (int)( mll/12 )
                    # guess a midi note
                    tmp_note = (int)( 12*mel_oct + pc)
                    # put guess in lower octave until it's under the mll
                    while tmp_note > mll:
                        tmp_note -= 12
                    tmp_midis.append( tmp_note )
                    to_delete = np.where( tmp_gct_pcs==pc )[0][0]
                    tmp_gct_pcs = np.delete(tmp_gct_pcs, to_delete)
                cs.chord = m21.chord.Chord( tmp_midis )
                tmp_idx += 1
        # we then have to make intermediate voices
        m = make_harmonisation_stream(self.m, self.idiom)
        return m
    # end apply_bbvl

# nn vl functions =======================================================
def chord_segments_to_NN_VL_conditions(s):
    # melody skyline - keep the lowest note of the melody for each segment
    melody_bin = np.zeros( (128, len(s)) )
    # matrix of available (binary) options for placing notes (upper-bounded
    # by melody notes)
    harmony_options_bin = np.zeros( (128, len(s)) )
    # keep the gct completion list of list, where we keep count of which gct
    # pitches have been covered and which ones remain to be covered as a binary
    # matrix, 1s for each gct pitch, that all need to become 0
    gct_completion = np.zeros( (12, len(s)) )
    # for each segment fill the respective matrices
    for i in range( len( s ) ):
        c = s[i]
        # get lowest melody note
        lmn = min( c.melody_midi )
        # just to make sure, check that lmn < 128
        if lmn >= 128:
            lmn = 127
        # assign it to the bin melody matrix
        melody_bin[ lmn , i ] = 1
        # get relative pitch classes from gct and tonality
        key = c.tonality.key
        gct = c.gct_chord
        rpcs = np.mod( gct[0]+key+gct[1:] , 12 )
        rpcs = rpcs.astype( int )
        # fill completion matrix
        gct_completion[ rpcs , i ] = 1
        for mult in range( 12 ):
            for rpc in rpcs:
                # get note in proper octave
                tmp_note = mult*12 + rpc
                # check if note exceeds limits imposed by melody note (upper limit)
                if tmp_note < lmn:
                    # assign note to matrix
                    harmony_options_bin[ tmp_note , i ] = 1
    return melody_bin, harmony_options_bin, gct_completion
# end chord_segments_to_NN_VL_conditions

'''
def apply_NN_VL_to_segments(s):
    # make matrices
    melody_bin, harmony_options_bin, gct_completion = chord_segments_to_NN_VL_conditions(s)
    # load model
    model = nnvl.PolyFiller( melody_bin, harmony_options_bin, gct_completion )
    # run NN VL
    model.run_NN_VL()
    # return matrix
    mm = model.matrix
    return mm
# end apply_NN_VL_to_segments

def apply_NN_voice_leading(m, idiom):
    # concatenate all chord segments from all phrases
    chord_segments = []
    for p in m.phrases:
        chord_segments.extend( p.melody_chord_segments )
    # __NN_VL__
    matrix = apply_NN_VL_to_segments(chord_segments)
    # keep a chord_segments copy to return
    # s = copy.deepcopy( chord_segments )
    # __NN_VL__
    tmp_idx = 0
    for p in m.phrases:
        # get tonality key
        key = p.tonality.key
        # for each chord segment get gct chord
        for cs in p.melody_chord_segments:
            chord = np.nonzero( matrix[ : , tmp_idx ] )
            chord_final = chord[0].tolist()
            cs.chord = m21.chord.Chord( chord_final )
            tmp_idx += 1
    m = make_harmonisation_stream(m, idiom)
    return m
    '''