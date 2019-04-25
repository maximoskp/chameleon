#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:49:44 2018

@author: maximoskaliakatsos-papakostas
"""

import os
cwd = os.getcwd()
import music21 as m21
import sys
# use folders of generation functions
sys.path.insert(0, cwd + '/CM_train')
import CM_TonalityGrouping_classes as tgc
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf
import copy

class MelodyInput:
    ''' information for an input melody '''
    def __init__(self, folderName, fileName):
        # 'metadata'
        self.name = fileName.split('.')[0]
        # harmonisation file name does not include extension,
        # to use it with several formats, e.g. xml, mid, txt (logging)
        self.harmonisation_file_name = []
        self.style = folderName.split('/')[-2]
        # parse music 21 piece
        p = m21.converter.parse(folderName+fileName)
        # keep input stream structure for re-building the harmonised melody
        self.input_stream = p
        # initialise output stream to be filled in voice leading
        self.output_stream = []
        # get necessary information from piece
        # tonality
        tonPart = p.parts[-2]
        tonChordified = tonPart.chordify()
        # grouping
        groupPart = p.parts[-1]
        groupChordified = groupPart.chordify()
        # chord constraints
        constrPart = p.parts[-3]
        constrChordified = constrPart.chordify()
        # harmonic rhythm
        harmRhythmPart = p.parts[2]
        harmRhythmChordified = harmRhythmPart.chordify()
        # important notes
        importantNotesPart = p.parts[1]
        importantNotesChordified = importantNotesPart.chordify()
        # melody
        melodyPart = p.parts[0]
        # flat all
        tonFlat = tonChordified.flat.getElementsByClass('Chord')
        groupFlat = groupChordified.flat.getElementsByClass('Chord')
        # keep flats of constraints, harmonic rhythm and important notes as they are
        self.constraints = constrChordified.flat.getElementsByClass('Chord')
        self.harmonicRhythm = harmRhythmChordified.flat.getElementsByClass('Chord')
        self.importantNotes = importantNotesChordified.flat.getElementsByClass('Chord')
        # we need melody notes flat for isolating note segments per chord
        self.melodyNotes = melodyPart.flat.notes
        # list of tonality objects
        if len(tonFlat) > 0:
            tmpTons = [tgc.Tonality(m21_chord_in=t) for t in tonFlat]
        else:
            tmpTons = [tgc.Tonality()]
        self.tonalities = tmpTons
        # list of grouping objects
        if len(groupFlat) > 0:
            tmpGroups = [tgc.Grouping(m21_chord_in=t) for t in groupFlat]
        else:
            tmpGroups = [tgc.Tonality()]
        self.groupings = tmpGroups
        # get offsets of all tonalities and groupings
        self.tonality_offsets = maf.get_offsets(self.tonalities)
        self.grouping_offsets = maf.get_offsets(self.groupings)
        # the first grouping or tonality should be in the beginning,
        # regardless of the xml info provided by the user,
        # to avoid empty phrases in the beginning
        if len(self.tonality_offsets) > 0:
            self.tonality_offsets[0] = 0
        if len(self.grouping_offsets) > 0:
            self.grouping_offsets[0] = 0
        self.phrases = self.make_phrase_structure()
    # end constructor
    
    def make_phrase_structure(self):
        phrases = []
        # initialisation - check if first tonality is indeed in the beginning
        ton_idx = 0
        phr_idx = 0
        # use index for printing logger images
        phrase_idx = 0
        curr_tonality = self.tonalities[ton_idx]
        curr_grouping = self.groupings[ (phr_idx+1)%len(self.groupings) ]
        all_offsets = sorted(list( set( sorted(self.tonality_offsets+self.grouping_offsets) ) ))
        curr_offset = 0
        # global "horisontal" counter
        offset_idx = 0
        curr_offset = all_offsets[offset_idx]
        offset_idx = 1
        while offset_idx < len(all_offsets):
            next_offset = all_offsets[offset_idx]
            curr_tonality = self.tonalities[ton_idx]
            curr_grouping = self.groupings[ (phr_idx+1)%len(self.groupings) ]
            # decide about the type of the phrase
            tmp_type = 'grouping'
            if curr_offset in self.tonality_offsets:
                tmp_type = 'tonality'
            # check if the next offset is a phrase offset
            if next_offset in self.grouping_offsets:
                phr_idx += 1
            # check if the next offset is a tonality offset
            if next_offset in self.tonality_offsets:
                ton_idx += 1
            # get a deep copy of everything that needs to be sliced
            # constraints
            tmp_allConstraints = copy.deepcopy( self.constraints )
            tmpConstraints = tmp_allConstraints.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # harmonic rhythm
            tmp_allHarmonicRhythm = copy.deepcopy( self.harmonicRhythm )
            tmpHarmonicRhythm = tmp_allHarmonicRhythm.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # important notes
            tmp_allImportantNotes = copy.deepcopy( self.importantNotes )
            tmpImportantNotes = tmp_allImportantNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # melody
            tmp_allNotes = copy.deepcopy( self.melodyNotes )
            tmpNotes = tmp_allNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            if len(tmpHarmonicRhythm) > 0:
                phrases.append( tgc.MelodyPhrase( curr_tonality, tmpConstraints, tmpHarmonicRhythm, tmpImportantNotes, tmpNotes, tmp_type, curr_grouping.level, next_offset, phrase_idx=phrase_idx ) )
                phrase_idx += 1
            curr_offset = next_offset
            offset_idx += 1
        # END WHILE
        # get the remaining chords, after the end of the last annotated tonality/phrase
        curr_tonality = self.tonalities[ton_idx]
        curr_grouping = self.groupings[ (phr_idx+1)%len(self.groupings) ]
        #self.chordsFlat.show('t')
        next_offset = self.melodyNotes[-1].offset + 1.0
        # get a deep copy of everything that needs to be sliced
        # constraints
        tmp_allConstraints = copy.deepcopy( self.constraints )
        tmpConstraints = tmp_allConstraints.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # harmonic rhythm
        tmp_allHarmonicRhythm = copy.deepcopy( self.harmonicRhythm )
        tmpHarmonicRhythm = tmp_allHarmonicRhythm.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # important notes
        tmp_allImportantNotes = copy.deepcopy( self.importantNotes )
        tmpImportantNotes = tmp_allImportantNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # melody
        tmp_allNotes = copy.deepcopy( self.melodyNotes )
        tmpNotes = tmp_allNotes.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # final phrase type is always grouping
        tmp_type = 'grouping'
        phrases.append( tgc.MelodyPhrase( curr_tonality, tmpConstraints, tmpHarmonicRhythm, tmpImportantNotes, tmpNotes, tmp_type, curr_grouping.level, next_offset, phrase_idx=phrase_idx ) )
        phrase_idx += 1
        return phrases