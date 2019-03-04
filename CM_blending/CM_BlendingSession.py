#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:16:46 2018

@author: maximoskaliakatsos-papakostas
"""

# import pickle
import os
cwd = os.getcwd()
import numpy as np
import copy
import pickle
import CM_ModeShiftingFunctions as msf
import CM_Transition_class as trn
import CM_BlendedIdiomCompiler as bic
import sys
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf

class BlendingSession:
    def __init__(self, id1, id2, md1=0, md2=0, grp1=True, grp2=True, ton_diff=0):
        self.idiom1 = id1
        self.idiom2 = id2
        # get modes for each idiom and 
        mode1, mode1_correct = self.get_mode_from_idiom(id1, md1)
        mode2, mode2_correct = self.get_mode_from_idiom(id2, md2)
        # after that, idiom is represented by its mode
        if not (mode1_correct and mode2_correct):
            print('Incorrect modes given: ', md1, ' - ', md2)
            return -1, -1
        # parallel tonality shift of the second mode if necessary
        if ton_diff != 0:
            mode2 = msf.parallel_shift_mode(mode2,ton_diff)
        self.mode1 = mode1
        self.mode2 = mode2
        # also keep idiom names for making the blend name
        self.name_1 = id1.name;
        self.name_2 = id2.name;
        # here we will need to shift one tonality
        # make two lists of the most usual N transitions for each idiom
        self.transitions_1_np = self.get_most_common_transitions(mode1, grp1)
        self.transitions_2_np = self.get_most_common_transitions(mode2, grp2)
        # constuct chord dictionary
        self.construct_chord_dictionary()
        # construct ontologies of selected transitions
        self.trans_ontologies_1 = self.make_transitions_ontology( self.transitions_1_np )
        self.trans_ontologies_2 = self.make_transitions_ontology( self.transitions_2_np )
        # compute property priorities for transitions in ontologies
        self.trans_ontologies_1 = self.compute_priorities( self.trans_ontologies_1, self.trans_ontologies_2 )
        self.trans_ontologies_2 = self.compute_priorities( self.trans_ontologies_2, self.trans_ontologies_1 )
        # construct all blending diamonds - input1, input2, generic, blends
        self.blending_diamonds = self.construct_diamonds()
        # filter chords in dictionary according to generic space constraints
        # also rate blends along the process
        self.blending_diamonds = self.generic_filter_and_rate()
        # run through each diamond and keep a total of best blends
        self.best_ab_blends, self.best_ba_blends = self.select_best_blends()
        # construct blended idiom
        self.blended_idiom = bic.compile_blended_idiom(self.mode1, self.mode2, ton_diff, self.best_ab_blends, self.best_ba_blends)
    # end constructor

    def select_best_blends(self):
        print('selecting best blends')
        # list of best blended transitions and their rates
        best_ab_blends = []
        best_ab_scores = []
        best_ba_blends = []
        best_ba_scores = []
        # iterate through diamonds and keep the single best blend from each
        for d in self.blending_diamonds:
            # get maximum score index
            # A->B chains =======================================================
            if len( d['ab_scores'] ) > 0:
                idx = np.argmax( d['ab_scores'] )
                best_ab_blends.append( d['ab_chains'][idx] )
                best_ab_scores.append( d['ab_scores'][idx] )
            # B->A chains =======================================================
            if len( d['ba_scores'] ) > 0:
                idx = np.argmax( d['ba_scores'] )
                best_ba_blends.append( d['ba_chains'][idx] )
                best_ba_scores.append( d['ba_scores'][idx] )
        # sort blends based on their scores
        # A->B chains =======================================================
        best_ab_scores = np.array( best_ab_scores )
        idxs_ab = np.argsort( -best_ab_scores )
        best_ab_scores = best_ab_scores[idxs_ab]
        best_ab_blends = [best_ab_blends[i] for i in idxs_ab]
        # B-> chains =======================================================
        best_ba_scores = np.array( best_ba_scores )
        idxs_ba = np.argsort( -best_ba_scores )
        best_ba_scores = best_ba_scores[idxs_ba]
        best_ba_blends = [best_ba_blends[i] for i in idxs_ba]
        return best_ab_blends, best_ba_blends
    # end select_best_blends

    def generic_filter_and_rate(self):
        # iterate over all generic spaces and return dictionary elements that comply with restrictions
        dd = 0
        for d in self.blending_diamonds:
            print('dd: ', dd)
            dd += 1
            # possible chains from a to b, which will be filtered and rated (as chains) at the time they get generated
            t_ab = []
            t_ba = []
            # keep all scores
            ab_scores = []
            ba_scores = []
            # construct all possible ab chains
            for c in self.possible_gcts_np:
                # A->B chains =====================================================================
                # form both components of the chain
                ab_1_transition = trn.Transition( d['input1'].from_chord_np['property'] , c )
                ab_2_transition = trn.Transition( c , d['input2'].to_chord_np['property'] )
                # check generic space of each component
                ab_1_gen_check = trn.check_generic_constraints( ab_1_transition , d['generic'] )
                ab_2_gen_check = trn.check_generic_constraints( ab_2_transition , d['generic'] )
                # keep only chains that have both transitions valid
                if ab_1_gen_check and ab_2_gen_check:
                    ab_1_transition = trn.rate_blend(ab_1_transition , d['input1'], d['input2'])
                    ab_2_transition = trn.rate_blend(ab_2_transition , d['input1'], d['input2'])
                    t_ab.append( [ab_1_transition, ab_2_transition] )
                    ab_scores.append( ab_1_transition.blending_score + ab_2_transition.blending_score )
                # B->A chains =====================================================================
                # form both components of the chain
                ba_1_transition = trn.Transition( d['input2'].from_chord_np['property'] , c )
                ba_2_transition = trn.Transition( c , d['input1'].to_chord_np['property'] )
                # check generic space of each component
                ba_1_gen_check = trn.check_generic_constraints( ba_1_transition , d['generic'] )
                ba_2_gen_check = trn.check_generic_constraints( ba_2_transition , d['generic'] )
                # keep only chains that have both transitions valid
                if ba_1_gen_check and ba_2_gen_check:
                    ba_1_transition = trn.rate_blend(ba_1_transition , d['input1'], d['input2'])
                    ba_2_transition = trn.rate_blend(ba_2_transition , d['input1'], d['input2'])
                    t_ba.append( [ba_1_transition, ba_2_transition] )
                    ba_scores.append( ba_1_transition.blending_score + ba_2_transition.blending_score )
            # append to the diamond structure
            print('--------- ab: ', len(t_ab))
            print('--------- ba: ', len(t_ba))
            d['ab_chains'] = t_ab
            d['ab_scores'] = ab_scores
            d['ba_chains'] = t_ba
            d['ba_scores'] = ba_scores
        return self.blending_diamonds
    # end generic_filter_and_rate
    def construct_diamonds(self):
        diamonds = []
        for t1 in self.trans_ontologies_1:
            for t2 in self.trans_ontologies_2:
                g = trn.compute_generic_space(t1, t2)
                # a chain has two transitions, one that goes form space X to new and one from new to space Y
                diamonds.append( {'input1': t1, 'input2': t2, 'generic': g, 'ab_chains': [], 'ba_chains': [], 'ab_scores': [], 'ba_scores': []} )
        return diamonds
    # end construct_triplets
    def make_transitions_ontology(self, t):
        ontos = []
        for i in range( len( t ) ):
            ontos.append( trn.Transition(t[i][0], t[i][1]) )
        return ontos
    # end make_transitions_ontology
    def compute_priorities(self, t1, t2):
        for i in range( len( t1 ) ):
            t1[i] = trn.compute_priorities( t1[i], t1, t2 )
        return t1
    # end compute_priorities

    def construct_chord_dictionary(self):
        # initialise with some standard types
        self.possible_types_np = [ np.array([0,4,7]), np.array([0,3,7]), np.array([0,3,6]), np.array([0,4,7,10]) ]
        # run through both idioms and pile up chord types
        for i in range( len( self.mode1.gct_info.gcts_labels ) ):
            tmp_type = maf.str2np( self.mode1.gct_info.gcts_labels[i] )[1:]
            if not any( np.array_equal(tmp_type, a_type) for a_type in self.possible_types_np ):
                self.possible_types_np.append( tmp_type )
        for i in range( len( self.mode2.gct_info.gcts_labels ) ):
            tmp_type = maf.str2np( self.mode2.gct_info.gcts_labels[i] )[1:]
            if not any( np.array_equal(tmp_type, a_type) for a_type in self.possible_types_np ):
                self.possible_types_np.append( tmp_type )
        # construct all possible chords (assign roots) for these possible types
        self.possible_gcts_np = []
        for t in self.possible_types_np:
            # check if type is a diminished and keep only roots 0,1,2
            if np.array_equal(t, np.array([0,3,6,9])):
                for i in range(3):
                    self.possible_gcts_np.append( np.append( i, t ) )
            elif np.array_equal(t, np.array([0,4,8])): # check if augmented and keep only 0,1,2,3
                for i in range(4):
                    self.possible_gcts_np.append( np.append( i, t ) )
            else: # assign all 12 roots
                for i in range(12):
                    self.possible_gcts_np.append( np.append( i, t ) )
    # end construct_chord_dictionary

    def get_mode_from_idiom(self, idiom, mode):
        # keep a control variable to check wether proper mode for idiom is given
        mode_correct = True
        mode_out = []
        # sort out what mode means and finally get the mode object for input idiom
        mode_keys = list( idiom.modes.keys() )
        # if mode is an index
        if isinstance(mode, int):
            # check if index larger than number of modes in idiom
            if mode >= len(mode_keys):
                mode_correct = False
            else:
                mode_out = idiom.modes[ mode_keys[ mode ] ]
        elif isinstance(mode, str):
            if mode not in mode_keys:
                mode_correct = False
            else:
                mode_out = idiom.modes[ mode ]
        else:
            mode_correct = False
        return mode_out, mode_correct
    # end get_mode_from_idiom

    def get_most_common_transitions(self, mode, grouping, nTrans=5):
        # get the proper gct info from mode and chords
        gct_info = mode.gct_info
        if grouping:
            gct_info = mode.gct_group_info
        # get markov matrix
        tr = copy.deepcopy(gct_info.gcts_markov)
        # get chord occurances
        occs = gct_info.gcts_occurances
        # multiply each row with the respective occurances
        for i in range(tr.shape[0]):
            tr[i,:] = occs[i]*tr[i,:]
            # zero-out repetitions
            tr[i,i] = 0
        # sort all elements in tr and get sorted indexes
        s_idxs = np.unravel_index( np.argsort( np.ravel( tr ) )[::-1] , (tr.shape[0],tr.shape[1]) )
        # return transitions as a list of gct pairs
        gcts = gct_info.gcts_labels
        trans_out = []
        nTrans = min( [nTrans, len(gcts)] )
        for i in range( nTrans ):
            trans_out.append( [ maf.str2np( gcts[ s_idxs[0][i] ] ) , maf.str2np( gcts[ s_idxs[1][i] ] ) ] )
        return trans_out
    # end get_most_common_transitions
# end BlendingSession

def blend_two_idioms_all_modes( idiom1_name, idiom2_name, saving_folder='blended_idioms/' ):
    idiomFolder = cwd
    with open(idiomFolder+'/trained_idioms/'+idiom1_name+'.pickle', 'rb') as handle:
        idiom1 = pickle.load(handle)
    with open(idiomFolder+'/trained_idioms/'+idiom2_name+'.pickle', 'rb') as handle:
        idiom2 = pickle.load(handle)
    # run through all modes of each idiom and blend all combinations in all TDs
    mode_keys_1 = list( idiom1.modes.keys() )
    mode_keys_2 = list( idiom2.modes.keys() )
    for i1 in range( len( mode_keys_1 ) ):
        for i2 in range( len( mode_keys_2 ) ):
            print('blending mode idxs: ', i1, ' - ', i2)
            # add all differences when finished testing
            for td in range(12):
                b = BlendingSession(idiom1, idiom2, md1=i1, md2=i2, ton_diff=td)
                idiom = b.blended_idiom
                # save blended idiom
                with open(saving_folder+idiom.name+'.pickle', 'wb') as handle:
                    pickle.dump(idiom, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end blend_two_idioms_all_modes

def blend_all_idioms_from_list( idioms_list ):
    for i1 in idioms_list:
        for i2 in idioms_list:
            blend_two_idioms_all_modes( i1, i2 )
# end blend_all_idioms_from_list