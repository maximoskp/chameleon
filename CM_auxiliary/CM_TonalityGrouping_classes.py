#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:04:32 2018

@author: maximoskaliakatsos-papakostas

"""

'''
Classes for:
- Tonality
- Grouping
- Phrases

Functions for:
- getting offsets of tonality and grouping objects
'''

# import music21 as m21
import numpy as np
import GCT_functions as gct
import CM_Misc_Aux_functions as maf
import copy
import time
import music21 as m21

class Tonality:
    ''' information about tonality '''
    def __init__(self, m21_chord_in=[]):
        # tonality is given as an m21 chord object
        if len(m21_chord_in) > 0:
            ton_midi_array = [p.pitch.midi for p in m21_chord_in]
            self.midi_tonality = np.array(ton_midi_array)
            self.key = ton_midi_array[0]%12
            self.mode = np.mod(self.midi_tonality - self.midi_tonality[0], 12)
            self.mode_name = maf.np2str( self.mode )
            self.mode_pcp = maf.array2relpcp( self.mode )
            self.condensed = np.insert(self.mode, 0, self.key)
            self.offset = m21_chord_in.offset
        else:
            ton_midi_array = range(60, 72, 1)
            self.midi_tonality = np.array(ton_midi_array)
            self.key = ton_midi_array[0]%12
            self.mode = np.mod(self.midi_tonality - self.midi_tonality[0], 12)
            self.mode_name = maf.np2str( self.mode )
            self.mode_pcp = maf.array2relpcp( self.mode )
            self.condensed = np.insert(self.mode, 0, self.key)
            self.offset = 0
# end Tonality class

class Grouping:
    ''' information about tonality '''
    def __init__(self, m21_chord_in=[]):
        # grouping is given as an m21 chord object
        if len(m21_chord_in) > 0:
            self.level = len(m21_chord_in)
            self.offset = m21_chord_in.offset
        else:
            self.level = 4
            self.offset = 0
# end Grouping class

class Cadence:
    ''' information about cadence '''
    def __init__(self, gcts_pair_in, level_in):
        # grouping is given as an m21 chord object
        self.level = level_in
        self.gcts_pair = gcts_pair_in
        self.label = maf.np2str(gcts_pair_in[0]) + "-" + maf.np2str(gcts_pair_in[1])
        self.penultimate_gct = maf.np2str(gcts_pair_in[0])
        self.penultimate_relative_pcp = maf.gct2relpcp(gcts_pair_in[0])
        self.final_gct = maf.np2str(gcts_pair_in[1])
        self.final_relative_pcp = maf.gct2relpcp(gcts_pair_in[1])
        self.probability = [] # this will be filled in later in mode
# end Cadence class

class TrainingPhrase:
    ''' information about training phrase '''
    def __init__(self, tonality_in, chords_in, type_in, level_in):
        # tonality is given as a tonality object of the class above
        # chords are given as m21 iterator of flat chords
        # type is a string: "tonality" or "phrase" showing if it ends
        #                              in tonality or grouping change
        # level is an integer showin the level of the phrase
        self.tonality = tonality_in
        self.type = type_in
        if type_in is 'tonality':
            self.level = 1
        else:
            self.level = level_in
        tmpChords = []
        for c in chords_in:
            tmpMidis = [p.midi for p in c.pitches]
            tmpChords.append( tmpMidis )
        self.midi_chords = tmpChords
        # make gcts
        tmpGCTs = []
        # auxiliary dictionary for storing gcts of respective midi chords
        self.midi_gct_dict = {}
        for c in self.midi_chords:
            tmpGCTs.append( gct.get_singe_GCT_of_chord(c, self.tonality.key, self.tonality.mode) )
            self.midi_gct_dict[str(c)] = tmpGCTs[-1]
        self.gct_chords = tmpGCTs
        if len(self.gct_chords) >= 2:
            self.cadence = Cadence(self.gct_chords[-2:], self.level)
        else:
            self.cadence = Cadence([self.gct_chords[-1], self.gct_chords[-1]], self.level)
        # GCT_VL
        self.gct_vl = self.make_gct_vl()
    # end constructor
    def make_gct_vl(self):
        # initialise an empty gct_vl dictionary
        gct_vl = {}
        prev_chord = []
        all_keys = list( self.midi_gct_dict.keys() )
        for i in range( len(all_keys) ):
            k = all_keys[i]
            # get back midi chord
            tmp_midi = np.fromstring(k[1:-1], sep=',').astype(int)
            # get rPCP of lowest note
            tmp_bass_rpcp = np.mod( np.min( tmp_midi ) - self.tonality.key , 12 )
            # get rpcp of gct
            tmp_gct = self.midi_gct_dict[k]
            tmp_gct_rpcp = np.mod( tmp_gct[0]+tmp_gct[1:] , 12 )
            # make the addition binary vector for the bass rpc
            inversions_bin = np.zeros(len(tmp_gct) - 1)
            bass_idx = np.argwhere(tmp_gct_rpcp == tmp_bass_rpcp)[0][0]
            inversions_bin[ bass_idx ] += 1
            # to-transitions binary
            to_transitions_bin = np.zeros( ( 25 , len(tmp_gct) - 1 ) ) # 25 will need to change to VL
            if i < len(all_keys)-1:
                # get next midi chord
                next_midi = np.fromstring(all_keys[i+1][1:-1], sep=',').astype(int)
                # get bass note of next chord
                next_bass = np.min( next_midi )
                # compute step to next note
                to_step = next_bass - np.min( tmp_midi )
                # bring the step value in the values of the to_transitions_bin
                to_step = (int)( to_step + (to_transitions_bin.shape[0]-1)/2 )
                if to_step < 0:
                    to_step = 0
                if to_step > to_transitions_bin.shape[0]-1:
                    to_step = to_transitions_bin.shape[0]-1
                # assign value to matrix
                to_transitions_bin[to_step, bass_idx] += 1
            # from-transitions binary
            from_transitions_bin = np.zeros( ( 25 , len(tmp_gct) - 1 ) ) # 25 will need to change to VL
            if i > 0:
                # get previous midi chord
                prev_midi = np.fromstring(all_keys[i-1][1:-1], sep=',').astype(int)
                # get bass note of previous chord
                prev_bass = np.min( prev_midi )
                # compute step to previous note
                from_step = prev_bass - np.min( tmp_midi )
                # bring the step value in the values of the from_transitions_bin
                from_step = (int)( from_step + (from_transitions_bin.shape[0]-1)/2 )
                if from_step < 0:
                    from_step = 0
                if from_step > from_transitions_bin.shape[0]-1:
                    from_step = from_transitions_bin.shape[0]-1
                # assign value to matrix
                from_transitions_bin[from_step, bass_idx] += 1
            # check if there is a key for this gct
            tmp_key = maf.np2str(tmp_gct)
            if tmp_key in gct_vl.keys():
                # add to the existing values
                gct_vl[tmp_key]['inversions'] = gct_vl[tmp_key]['inversions'] + inversions_bin
                gct_vl[tmp_key]['mel2bass'].append( np.max(tmp_midi)-np.min(tmp_midi) )
                gct_vl[tmp_key]['to_bvl'] = gct_vl[tmp_key]['to_bvl'] + to_transitions_bin
                gct_vl[tmp_key]['from_bvl'] = gct_vl[tmp_key]['from_bvl'] + from_transitions_bin
            else:
                # create a new entry with this value
                gct_vl[tmp_key] = {
                    'inversions':inversions_bin,
                    'mel2bass':[np.max(tmp_midi)-np.min(tmp_midi)],
                    'to_bvl':to_transitions_bin,
                    'from_bvl':from_transitions_bin
                }
        return gct_vl
# end TrainingPhrase class

class MelodyChordSegment:
    ''' a segment of harmonic rhythm in input melody '''
    # end add_constraint
    def __init__(self, constraints_in, importantNotes_in, melody_in, tonality_in):
        self.is_constraint = False
        self.user_constraint = False
        # the following will need to be filled in harmonisation
        self.gct_chord = [] # this is filled below if the chord is constraint
        self.gct_label = []
        self.midi_chord = [] # this is filled by voice leading
        self.gct_rpcp = []
        self.melody = melody_in
        self.tonality = tonality_in
        # check if constraints_in is not empty and fill in constraint
        if len(constraints_in) > 0:
            self.gct_chord = constraints_in
            self.gct_label = maf.np2str( constraints_in )
            self.gct_rpcp = maf.gct2relpcp( self.gct_chord )
            self.is_constraint = True
            self.user_constraint = True
        # get melody midi pitches
        tmp_melody_midi = []
        for m in melody_in:
            tmp_melody_midi.append( m.pitch.midi )
        self.melody_midi = np.array(tmp_melody_midi)
        # get melody relative pitch classes
        self.relative_pcs = np.mod( self.melody_midi - self.tonality.key, 12 )
        # get relative pcp
        self.relative_pcp = np.histogram( self.relative_pcs, bins=range(13) )[0]
        # get important note midis
        tmp_important_midi = []
        melody_copy = copy.deepcopy( melody_in )
        for m in importantNotes_in:
            n = melody_copy.getElementsByOffset(m.offset).getElementsByClass('Note')
            if n:
                tmp_important_midi.append( n[0].pitch.midi )
        self.important_melody_midi = np.array(tmp_important_midi)
        # get melody important relative pitch classes
        self.important_relative_pcs = np.mod( self.important_melody_midi - self.tonality.key, 12 )
        # get relative pcp
        self.important_relative_pcp = np.histogram( self.important_relative_pcs, bins=range(13) )[0]
    # end constructor
# end MelodyChordSegment

class MelodyPhrase:
    ''' information about melody phrase '''
    def __init__(self, tonality_in, constraints_in, harmonicRhythm_in, importantNotes_in, melody_in, type_in, level_in, ending_offset):
        # tonality is given as a tonality object of the class above
        # constraints_in, harmonicRhythm_in and importantNotes_in are given as m21 iterator of flat chords
        # melody_in is given as m21 iterator of notes
        # type is a string: "tonality" or "phrase" showing if it ends
        #                              in tonality or grouping change
        # level is an integer showin the level of the phrase
        # ending_offset is the offset number where the next phrase begins - final phrase has an arbitrarily expanded offset (e.g. + 1.0)
        self.tonality = tonality_in
        self.idiom_mode_label = [] # to be filled in cHMM, when best matching mode is decided
        self.constraints = constraints_in
        self.harmonicRhythm = harmonicRhythm_in
        self.importantNotes = importantNotes_in
        self.melody = melody_in
        self.type = type_in
        self.level = level_in
        # initialise empty chord segment
        self.melody_chord_segments = []
        # get chord offsets to slice chords parts in phrase
        self.chord_offsets = maf.get_offsets(self.harmonicRhythm)
        # for each chord segment, isolate melody information (notes, important notes constraints)
        curr_offset = self.chord_offsets[0]
        offset_idx = 0
        # TODO: we need a combined array of offsets from harmonic rhythm offsets and constraints - see TODOs.txt -> Melodic input -> point 1
        while offset_idx < (len(self.chord_offsets) - 1):
            offset_idx += 1
            next_offset = self.chord_offsets[offset_idx]
            # get parts to be filled in the chord segment
            # constraints
            tmp_allConstraints = copy.deepcopy( self.constraints )
            tmpConstraints = tmp_allConstraints.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # important notes
            tmp_allImportantNotes = copy.deepcopy( self.importantNotes )
            tmpImportantNotes = tmp_allImportantNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # melody
            tmp_allNotes = copy.deepcopy( self.melody )
            tmpNotes = tmp_allNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # constraints need to be sent as GCTs - not m21 chords
            tmpGCTconstraints = []
            if tmpConstraints:
                tmpGCTconstraints = gct.get_singe_GCT_of_m21chord(tmpConstraints, self.tonality.key, self.tonality.mode)
            self.melody_chord_segments.append( MelodyChordSegment(tmpGCTconstraints , tmpImportantNotes, tmpNotes, self.tonality ) )
            curr_offset = next_offset
        # get the remaining last chord segment
        next_offset = self.chord_offsets[-1] + ending_offset
        # get parts to be filled in the chord segment
        # constraints
        tmp_allConstraints = copy.deepcopy( self.constraints )
        tmpConstraints = tmp_allConstraints.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # important notes
        tmp_allImportantNotes = copy.deepcopy( self.importantNotes )
        tmpImportantNotes = tmp_allImportantNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # melody
        tmp_allNotes = copy.deepcopy( self.melody )
        tmpNotes = tmp_allNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # constraints need to be sent as GCTs - not m21 chords
        tmpGCTconstraints = []
        if tmpConstraints:
            tmpGCTconstraints = gct.get_singe_GCT_of_m21chord(tmpConstraints, self.tonality.key, self.tonality.mode)
        self.melody_chord_segments.append( MelodyChordSegment(tmpGCTconstraints , tmpImportantNotes, tmpNotes, self.tonality ) )
        # check if there are chord segments with no melodic notes and
        # take either the last note of the previous segment
        # or the first note of the next
        for i in range( len( self.melody_chord_segments ) ):
            if not self.melody_chord_segments[i].melody:
                # check if it's the first segment that misses melody
                if i == 0:
                    # check if it's also the final chord - single chord phrase
                    if i == len( self.melody_chord_segments )-1:
                        # assign melody note based on tonality
                        self.melody_chord_segments[i].melody = m21.note.Note( self.tonality.key + 72 )
                        self.melody_chord_segments[i].melody_midi = self.tonality.key + 72
                        self.melody_chord_segments[i].relative_pcs = self.tonality.key
                        self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                    else:
                        # check if next chord doesn't have melody
                        if not self.melody_chord_segments[i+1].melody:
                            # assign melody note based on tonality
                            self.melody_chord_segments[i].melody = m21.note.Note( self.tonality.key + 72 )
                            self.melody_chord_segments[i].melody_midi = self.tonality.key + 72
                            self.melody_chord_segments[i].relative_pcs = self.tonality.key
                            self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                        else:
                            # get first melody note of next segment
                            self.melody_chord_segments[i].melody = m21.note.Note( self.melody_chord_segments[i+1].melody_midi[0] )
                            self.melody_chord_segments[i].melody_midi = self.melody_chord_segments[i+1].melody_midi[0]
                            self.melody_chord_segments[i].relative_pcs = self.melody_chord_segments[i+1].relative_pcs
                            self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                else:
                    # check if it's the final segment
                    if i == len( self.melody_chord_segments )-1:
                        # check if previous segment doesn't have note
                        if not self.melody_chord_segments[i-1].melody:
                            # assign melody note based on tonality
                            self.melody_chord_segments[i].melody = m21.note.Note( self.tonality.key + 72 )
                            self.melody_chord_segments[i].melody_midi = self.tonality.key + 72
                            self.melody_chord_segments[i].relative_pcs = self.tonality.key
                            self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                        else:
                            # get last melody note of previous segment
                            self.melody_chord_segments[i].melody = m21.note.Note( self.melody_chord_segments[i-1].melody_midi[0] )
                            self.melody_chord_segments[i].melody_midi = self.melody_chord_segments[i-1].melody_midi[0]
                            self.melody_chord_segments[i].relative_pcs = self.melody_chord_segments[i-1].relative_pcs
                            self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                    else:
                        # if it's not the final segment, first check previous doesn't have notes
                        if not self.melody_chord_segments[i-1].melody:
                            # then check if next doesn't have as well
                            if not self.melody_chord_segments[i+1].melody:
                                # assign melody note based on tonality
                                self.melody_chord_segments[i].melody = m21.note.Note( self.tonality.key + 72 )
                                self.melody_chord_segments[i].melody_midi = self.tonality.key + 72
                                self.melody_chord_segments[i].relative_pcs = self.tonality.key
                                self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                            else:
                                # get first melody note of next segment
                                self.melody_chord_segments[i].melody = m21.note.Note( self.melody_chord_segments[i+1].melody_midi[0] )
                                self.melody_chord_segments[i].melody_midi = self.melody_chord_segments[i+1].melody_midi[0]
                                self.melody_chord_segments[i].relative_pcs = self.melody_chord_segments[i+1].relative_pcs
                                self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
                        else:
                            # get last melody note of previous segment
                            self.melody_chord_segments[i].melody = m21.note.Note( self.melody_chord_segments[i-1].melody_midi[0] )
                            self.melody_chord_segments[i].melody_midi = self.melody_chord_segments[i-1].melody_midi[0]
                            self.melody_chord_segments[i].relative_pcs = self.melody_chord_segments[i-1].relative_pcs
                            self.melody_chord_segments[i].relative_pcp = np.histogram( self.melody_chord_segments[i].relative_pcs, bins=range(13) )[0]
# end MelodyPhrase class