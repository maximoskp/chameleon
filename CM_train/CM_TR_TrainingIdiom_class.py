#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:04:32 2018

@author: maximoskaliakatsos-papakostas
"""

from collections import Counter
import sys
import os
cwd = os.getcwd()
import glob
import CM_TR_TrainingPiece_class as tpc
import numpy as np
import scipy.stats as scp
import copy
# import matplotlib.pyplot as plt
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf

class GCT_info:
    ''' information about GCTs and GCTgroups in a mode '''
    def __init__(self):
        self.gcts_array = [] # GCTgroups don't have it
        self.gcts_counter = [] # GCTgroups don't have it
        self.gct_group_structures = [] # only GCTgroups have it
        self.gcts_membership_dictionary = {} # only GCTgroups have it
        self.gcts_labels = []
        self.gcts_relative_pcs = []
        self.gcts_occurances = [] # __NEW__
        self.gcts_probabilities = [] # __NEW__ see where this is computed and compute occurances
        self.gcts_initial_array = [] # GCTgroups don't need it
        self.gcts_initial_counter = [] # GCTgroups don't have it
        self.gcts_initial_probabilities = []
        self.gcts_transitions_sum = [] # GCTgroups don't need it
        self.gcts_markov = []
        self.gct_vl_phrases = [] # gather vl stats for all phrases
        self.gct_vl_dict = {} # a dictionary with keys GCT labels and values with the aggregated stats
        self.gct_vl = [] # an array with only the agreegated stats, ordered per label sequence

class TrainingCadences:
    ''' information about cadences in a mode '''
    def __init__(self, idiom_name_in, mode_name_in, cadence_type_in):
        self.idiom_name = idiom_name_in
        self.mode_name = mode_name_in
        self.cadence_type = cadence_type_in
        self.all_cadence_labels = [] # all found cadences labels (doubles)
        self.all_cadence_structures = []
        self.cadences_counter = [] # the counter object that will help build the rest
        self.cadence_labels = [] # unique cadence labels
        self.cadences_dictionary = {} # keys are labels, values are cadence structures, along with probs
        # we don't need the following
        '''
        self.penultimate_gct = []
        self.penultimate_relative_pcp = []
        self.final_gct = []
        self.final_relative_pcp = []
        '''
        self.cadence_occurances = []
        self.cadence_probabilities = []
    # end constructor
    def append_cadence_label(self, label_in):
        self.all_cadence_labels.append(label_in)
    # end append_cadence_pair
    def append_cadence_structure(self, cadence_in):
        self.all_cadence_structures.append(cadence_in)
    # end append_cadence_structure
    def retrieve_cadence_structure_by_label(self, label_in):
        r = []
        for s in self.all_cadence_structures:
            if s.label == label_in:
                r = s;
                break;
        if not r:
            print('cadence not found! - ', label_in, ' - that is weird... in CM_TR_TrainingIdiom_class')
        return r
    # end retrieve_cadence_structure_by_label
    def make_cadences_stats(self):
        # make stats of collected cadences
        print('making cadence stats')
        self.cadences_counter = Counter(self.all_cadence_labels)
        # make labels based on counter
        self.cadence_labels = list( set( self.cadences_counter ) )
        # occurances per label
        occs = []
        for i in range( len( self.cadence_labels) ):
            occs.append( self.cadences_counter[ self.cadence_labels[i] ] )
        self.cadence_occurances = np.array( occs )
        # probabilities
        if np.sum(self.cadence_occurances) > 0:
            self.cadence_probabilities = self.cadence_occurances/np.sum(self.cadence_occurances)
        else:
            self.cadence_probabilities = 0
        # isolate unique structures according to label
        for i in range( len( self.cadence_labels) ):
            self.cadences_dictionary[ self.cadence_labels[i] ] = self.retrieve_cadence_structure_by_label( self.cadence_labels[i] )
            self.cadences_dictionary[ self.cadence_labels[i] ].probability = self.cadence_probabilities[i]
    # end make_cadences_stats

class TrainingMode:
    ''' information about a mode in an idiom '''
    def __init__(self, idiom_name_in, mode_in):
        self.mode = mode_in
        self.mode_pcp = maf.array2relpcp( self.mode )
        self.mode_name = maf.np2str( self.mode )
        self.idiom_name = idiom_name_in
        # __NEW__ ----------------------------------
        # include this area in the GCT_info class/object
        self.gct_info = GCT_info()
        self.gct_group_info = GCT_info()
        # __NEW__-----------------------------------
        # initialise cadences
        self.cadences = {}
        self.cadences['intermediate'] = TrainingCadences(self.idiom_name, self.mode_name, 'intermediate')
        self.cadences['final'] = TrainingCadences(self.idiom_name, self.mode_name, 'final')
    # end constructor
    def append_gct_in_array(self, gct_in):
        tmp_string_array = maf.np2str(gct_in)
        self.gct_info.gcts_array.append( tmp_string_array )
    # end append_gct_in_array
    def append_initial_gct_in_array(self, gct_in):
        tmp_string_array = maf.np2str(gct_in)
        self.gct_info.gcts_initial_array.append( tmp_string_array )
    # end append_initial_gct_in_array
    def gct_label_to_rpc(self, l):
        # remove ' ', '[' and ']'
        # s = list( filter( lambda a: a!=' ' and a!='[' and a!=']', l ) )
        s = l.replace('[', '')
        s = s.replace(']', '')
        # split number strings
        s = s.split(" ")
        # to integer list
        x = list( map( int, s ) )
        # to numpy array
        n = np.array(x)
        # from condensed GCT to rpc
        r = np.mod( n[0]+n[1:] , 12)
        # to binary
        b = np.zeros(12)
        b[r] = 1
        return b
    # end gct_label_to_rpc
    def make_gct_relative_pcs(self):
        for l in self.gct_info.gcts_labels:
            # string label to np array
            self.gct_info.gcts_relative_pcs.append( self.gct_label_to_rpc(l) )
    # end make_gct_relative_pcs
    def make_gct_initial_probabilities(self):
        self.gct_info.gcts_initial_probabilities = np.zeros( len(self.gct_info.gcts_labels) )
        tmp_initial_labels = list( self.gct_info.gcts_initial_counter )
        for l in tmp_initial_labels:
            # string label to np array
            self.gct_info.gcts_initial_probabilities[ self.gct_info.gcts_labels.index( l ) ] = self.gct_info.gcts_initial_counter[ l ]
        # normalise probabilities
        if np.sum(self.gct_info.gcts_initial_probabilities) > 0:
            self.gct_info.gcts_initial_probabilities = self.gct_info.gcts_initial_probabilities/np.sum( self.gct_info.gcts_initial_probabilities )
    # end make_gct_initial_probabilities
    def smoothener(self, semitones):
        # smoothening normal
        x = np.linspace(scp.norm.ppf(0.01),scp.norm.ppf(0.99), 2*semitones+1)
        smooth = scp.norm.pdf(x)
        return smooth
    def process_vl_stats(self):
        # small value for zeros
        small_val = 0.0000001
        # for each element in self.gct_info.gct_vl_dict and self.gct_info.gct_vl
        for i,l in enumerate(self.gct_info.gcts_labels):
            stats = self.gct_info.gct_vl_dict[l]
            # inversions --------------------------------------------------------
            # add small value
            stats['inversions'][ stats['inversions']==0 ] = small_val
            # normalise to sum to 1
            if np.sum(stats['inversions']) != 0:
                stats['inversions'] = stats['inversions']/np.sum(stats['inversions'])
            # mel2bass: aggregate in a num_octaves range-------------------------
            num_octaves = 6  # 6 will need to change to VL
            tmp_distr = np.zeros(12*num_octaves)
            for n in stats['mel2bass']:
                if n >= 12*num_octaves:
                    n = 12*num_octaves-1
                tmp_distr[n] += 1
            # convolve
            tmp_distr = np.convolve(tmp_distr, self.smoothener(5), 'same')
            # add small value
            tmp_distr[ tmp_distr==0 ] = small_val
            # normalise to sum to 1
            if np.sum(tmp_distr) != 0:
                tmp_distr = tmp_distr/np.sum(tmp_distr)
            stats['mel2bass'] = tmp_distr
            # to bvl: smoothen and normalise for each column ------------------
            tmp_distr = stats['to_bvl']
            # for each row
            for j in range( tmp_distr.shape[1] ):
                # convolve
                tmp_distr[:,j] = np.convolve( tmp_distr[:,j] , self.smoothener(3) , 'same' )
                # add small value
                tmp_distr[ tmp_distr[:,j]==0 ,j] = small_val
                # normalise
                if np.sum(tmp_distr[:,j]) != 0:
                    tmp_distr[:,j] = tmp_distr[:,j]/np.sum(tmp_distr[:,j])
            stats['to_bvl'] = tmp_distr
            # from bvl: smoothen and normalise for each column ------------------
            tmp_distr = stats['from_bvl']
            # for each row
            for j in range( tmp_distr.shape[1] ):
                # convolve
                tmp_distr[:,j] = np.convolve( tmp_distr[:,j] , self.smoothener(3) , 'same' )
                # add small value
                tmp_distr[ tmp_distr[:,j]==0 ,j] = small_val
                # normalise
                if np.sum(tmp_distr[:,j]) != 0:
                    tmp_distr[:,j] = tmp_distr[:,j]/np.sum(tmp_distr[:,j])
            stats['from_bvl'] = tmp_distr
            # assign
            self.gct_info.gct_vl_dict[l] = stats
            self.gct_info.gct_vl[i] = stats
    # end process_vl_stats
    def make_gct_vl(self):
        for l in self.gct_info.gcts_labels:
            stats = {}
            # for each dictionary in every phrase
            vl_phrase_copy = copy.deepcopy( self.gct_info.gct_vl_phrases )
            for vl in vl_phrase_copy:
                # check if phrase dict (vl) has stats for this label (l)
                if l in vl.keys():
                    # get stats from dictionary corresponding to label
                    tmp_stats = vl[l]
                    # aggregate stats - if empty copy current stats
                    if not stats:
                        stats = tmp_stats
                    else:
                        for k in stats.keys():
                            stats[k] += tmp_stats[k]
            self.gct_info.gct_vl_dict[l] = stats
            self.gct_info.gct_vl.append( stats )
        self.process_vl_stats()
    # end make_gct_vl
    def make_gct_structures(self):
        # __NEW__ we need to put all those in the GCT_info object
        self.gct_info.gcts_counter = Counter( self.gct_info.gcts_array )
        self.gct_info.gcts_labels = list(self.gct_info.gcts_counter)
        # compute gct occurances
        for l in self.gct_info.gcts_labels:
            self.gct_info.gcts_occurances.append( self.gct_info.gcts_counter[l] )
        self.gct_info.gcts_occurances = np.array( self.gct_info.gcts_occurances )
        # and probabilities
        if np.sum( self.gct_info.gcts_occurances ) != 0:
            self.gct_info.gcts_probabilities = self.gct_info.gcts_occurances/np.sum( self.gct_info.gcts_occurances )
        else:
            self.gct_info.gcts_occurances = np.zeros( len(self.gct_info.gcts_occurances) )
        self.gct_info.gcts_initial_counter = Counter( self.gct_info.gcts_initial_array )
        self.make_gct_relative_pcs()
        self.make_gct_initial_probabilities()
        self.gct_info.gcts_transitions_sum = np.zeros( ( len(self.gct_info.gcts_labels), len(self.gct_info.gcts_labels) ) )
        self.gct_info.gcts_markov = np.zeros( ( len(self.gct_info.gcts_labels), len(self.gct_info.gcts_labels) ) )
        self.make_gct_vl()
        # TODO: here we could do GCT chord grouping
    # end make_gct_structures
    def add_gcts_to_transitions(self, gcts_in):
        # gcts is a list of gcts in np array format
        # gets gct sequence of a phrase and increases the respective transitions
        prev_gct = maf.np2str(gcts_in[0])
        for i in range(1, len(gcts_in), 1):
            next_gct = maf.np2str(gcts_in[i])
            # add to markov
            self.gct_info.gcts_transitions_sum[ self.gct_info.gcts_labels.index(prev_gct), self.gct_info.gcts_labels.index(next_gct) ] += 1
            prev_gct = next_gct
    # end add_gcts_to_markov
    def make_gcts_markov(self):
        for i in range(self.gct_info.gcts_transitions_sum.shape[0]):
            if np.sum(self.gct_info.gcts_transitions_sum[i,:]) > 0:
                self.gct_info.gcts_markov[i,:] = self.gct_info.gcts_transitions_sum[i,:]/np.sum(self.gct_info.gcts_transitions_sum[i,:])
    # end make_gcts_markov
    def plot_my_matrix(self):
        print('plotting deactivated for server')
        '''
        tmpfig = plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(self.gct_info.gcts_markov, cmap='gray_r', interpolation='none');
        plt.xticks(range(len(self.gct_info.gcts_labels)), self.gct_info.gcts_labels, rotation='vertical')
        plt.yticks(range(len(self.gct_info.gcts_labels)), self.gct_info.gcts_labels)
        tmpfig.savefig('figs/' + self.idiom_name + self.mode_name + '.png', format='png', dpi=300, bbox_inches="tight")
        plt.clf()
        '''
        # plt.show()
    # end plot_my_matrix
    def make_cadence_stats(self):
        for c in self.cadences.values():
            c.make_cadences_stats()
    # end make_cadence_stats

class TrainingIdiom:
    ''' information for a training idiom '''
    def __init__(self, folderName):
        # 'metadata'
        self.name = folderName.split('/')[-2]
        # dictionary of available modes in idiom
        self.modes = {}
        # get current directory for coming back
        cwd = os.getcwd()
        # visit the folder of the idiom
        os.chdir(folderName)
        # get all .xml files
        allDocs = glob.glob("*.xml")
        # get back
        # visit the folder of the idiom
        os.chdir(cwd)
        # check if folder is empty
        if len(allDocs) < 1:
            sys.exit("readAllXMLfiles.py: No XML files there!")
        # get all gcts from all phrases of all pieces
        tmp_all_phrases = []
        # for all pieces
        for pieceName in allDocs:
            print(pieceName)
            p = tpc.TrainingPiece(folderName, pieceName)
            # for all phrases in piece
            for phrase in p.phrases:
                # do not accept phrases that have only one gct - no transition
                if len(phrase.gct_chords) > 1:
                    # check if mode already exists, else create new
                    tmp_mode = maf.np2str( phrase.tonality.mode )
                    if tmp_mode not in self.modes:
                        self.modes[tmp_mode] = TrainingMode(self.name, phrase.tonality.mode)
                    # store phrase for further processing - markov
                    tmp_all_phrases.append(phrase)
                    # get first gct of phrase and store it in initial gcts
                    if len( phrase.gct_chords ) > 0:
                        self.modes[tmp_mode].append_initial_gct_in_array( phrase.gct_chords[0] )
                    # store gcts in phrases
                    for gct in phrase.gct_chords:
                        self.modes[tmp_mode].append_gct_in_array( gct )
                    # store entire gct_vl dictionary in phrase
                    self.modes[tmp_mode].gct_info.gct_vl_phrases.append( phrase.gct_vl )
                    # store cadences in phrases
                    if phrase.cadence.level > 2:
                        self.modes[tmp_mode].cadences['final'].append_cadence_label(phrase.cadence.label)
                        self.modes[tmp_mode].cadences['final'].append_cadence_structure(phrase.cadence)
                    else:
                        self.modes[tmp_mode].cadences['intermediate'].append_cadence_label(phrase.cadence.label)
                        self.modes[tmp_mode].cadences['intermediate'].append_cadence_structure(phrase.cadence)
        # after constructing all modes, aggregate gcts in all modes
        for m in self.modes.values():
            m.make_gct_structures()
        # run again through phrases and construct markov
        for phrase in tmp_all_phrases:
            tmp_mode = maf.np2str( phrase.tonality.mode )
            self.modes[tmp_mode].add_gcts_to_transitions( phrase.gct_chords )
        # after adding all gcts to transition sums, make markov probabilities
        for m in self.modes.values():
            m.make_gcts_markov()
            m.make_cadence_stats()
# end TrainingIdiom

class BlendingIdiom:
    ''' information for a blended idiom '''
    def __init__(self, name):
        # 'metadata'
        self.name = name
        # dictionary of available modes in idiom
        self.modes = {}