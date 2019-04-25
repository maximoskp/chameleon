#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:04:32 2018

@author: maximoskaliakatsos-papakostas
"""

import music21 as m21
import CM_TonalityGrouping_classes as tgc
import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf
import copy
# use folder of printing functions
sys.path.insert(0, cwd + '/CM_logging')
import harmonisation_printer as prt

class TrainingPiece:
    ''' information for a training piece '''
    def __init__(self, folderName, fileName, logging=False, log_file=[]):
        # 'metadata'
        self.name = fileName.split('.')[0]
        self.style = folderName.split('/')[-2]
        # parse music 21 piece
        p = m21.converter.parse(folderName+fileName)
        # get necessary information from piece
        # tonality
        tonPart = p.parts[-4]
        # check if expandable and expand
        if m21.repeat.Expander( tonPart ).isExpandable():
            tonPart = m21.repeat.Expander( tonPart ).process()
        tonChordified = tonPart.chordify()
        # grouping
        groupPart = p.parts[-3]
        # check if expandable and expand
        if m21.repeat.Expander( groupPart ).isExpandable():
            groupPart = m21.repeat.Expander( groupPart ).process()
        groupChordified = groupPart.chordify()
        # reduction
        r1 = p.parts[-1]
        r2 = p.parts[-2]
        # check if expandable and expand
        if m21.repeat.Expander( r1 ).isExpandable():
            r1 = m21.repeat.Expander( r1 ).process()
        if m21.repeat.Expander( r2 ).isExpandable():
            r2 = m21.repeat.Expander( r2 ).process()
        rc = m21.stream.Score()
        rc.insert(0, r1)
        rc.insert(0, r2)
        rcChordified = rc.chordify()
        # flat all
        tonFlat = tonChordified.flat.getElementsByClass('Chord')
        groupFlat = groupChordified.flat.getElementsByClass('Chord')
        chordsFlat = rcChordified.flat.getElementsByClass('Chord')
        # we need chord flats for isolating chord segments
        self.chordsFlat = chordsFlat
        # list of tonality objects
        if len(tonFlat) > 0:
            tmpTons = [tgc.Tonality(m21_chord_in=t) for t in tonFlat]
        else:
            tmpTons = [tgc.Tonality()]
        self.tonalities = tmpTons
        # list of grouping objects
        if len(groupFlat) > 0:
            tmpGroups = [tgc.Grouping(m21_chord_in=t) for t in groupFlat]
            self.shift_grouping_levels(tmpGroups)
        else:
            tmpGroups = [tgc.Grouping()]
        self.groupings = tmpGroups
        # list of midi chords
        tmpChords = []
        for c in chordsFlat:
            tmpMidis = [p.midi for p in c.pitches]
            tmpChords.append( tmpMidis )
        self.midi_chords = tmpChords
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
        # log tonality and grouping offsets
        if logging:
            tmp_log_line = 'tonality offsets: ' + str(self.tonality_offsets) + '\n'
            tmp_log_line += 'grouping offsets: ' + str(self.grouping_offsets)
            prt.print_log_line( log_file, tmp_log_line )
        self.phrases = self.make_phrase_structure(logging=logging, log_file=log_file)
    # end constructor
    def shift_grouping_levels(self, groups):
        # keep first level - lets assign 4 to the final
        # tmpFirst = groups[0].level
        for i in range( len( groups ) - 1 ):
            groups[i].level = groups[i+1].level
        # apply first to last
        # groups[-1].level = tmpFirst
        groups[-1].level = 4
        return groups
    # end shift_grouping_levels
    def make_phrase_structure(self, logging=False, log_file=[]):
        phrases = []
        # initialisation - check if first tonality is indeed in the beginning
        ton_idx = 0
        phr_idx = 0
        curr_tonality = self.tonalities[ton_idx]
        curr_grouping = self.groupings[phr_idx]
        all_offsets = sorted(list( set( sorted(self.tonality_offsets+self.grouping_offsets) ) ))
        curr_offset = 0
        # global "horisontal" counter
        offset_idx = 0
        if all_offsets[offset_idx] == 0:
            offset_idx = 1
        while offset_idx < len(all_offsets):
            next_offset = all_offsets[offset_idx]
            curr_tonality = self.tonalities[ton_idx]
            curr_grouping = self.groupings[phr_idx]
            tmp_type = 'grouping'
            if curr_offset in self.tonality_offsets:
                tmp_type = 'tonality'
            # check if the next offset is a phrase offset
            if next_offset in self.grouping_offsets:
                phr_idx += 1
            # check if the next offset is a tonality offset
            if next_offset in self.tonality_offsets:
                ton_idx += 1
            tmp_allChords = copy.deepcopy( self.chordsFlat )
            tmpChords = tmp_allChords.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
            # log phrase starting and ending offsets
            if logging:
                tmp_log_line = 'phrase starting offset: ' + str(curr_offset) + '\n'
                tmp_log_line += 'phrase ending offset: ' + str(next_offset)
                prt.print_log_line( log_file, tmp_log_line )
            # append phrase only if more than 2 chords are present
            if len( tmpChords ) >= 2:
                phrases.append( tgc.TrainingPhrase( curr_tonality, tmpChords, tmp_type, curr_grouping.level, logging=logging, log_file=log_file ) )
            curr_offset = next_offset
            offset_idx += 1
        # END WHILE
        # get the remaining chords, after the end of the last annotated tonality/phrase
        curr_tonality = self.tonalities[ton_idx]
        curr_grouping = self.groupings[phr_idx]
        #self.chordsFlat.show('t')
        next_offset = self.chordsFlat[-1].offset + 1.0
        tmp_allChords = copy.deepcopy( self.chordsFlat )
        tmpChords = tmp_allChords.getElementsByOffset(curr_offset, next_offset, includeEndBoundary=False)
        # final phrase type is always grouping
        tmp_type = 'grouping'
        if len( tmpChords ) >= 2:
            phrases.append( tgc.TrainingPhrase( curr_tonality, tmpChords, tmp_type, curr_grouping.level, logging=logging, log_file=log_file ) )
        return phrases