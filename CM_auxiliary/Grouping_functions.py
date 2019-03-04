#xml as input, returns an xml with gct harmonic analysis and comments (for any double gcts for the same chord)
# from music21 import *
# import itertools
# from itertools import combinations
import numpy as np

def str2np(s):
    return np.fromstring(s[1:-1], sep=' ')
# end str2np
def gct2relpcp(g):
    r_pc = np.mod(g[0] + g[1:], 12)
    return np.histogram( r_pc, bins=range(13) )[0]
# end gct2relpcp
def mode2relpcp(m):
    return np.histogram( m, bins=range(13) )[0]

class GCT_group:
    ''' attributes and methods concerning a GCT group '''
    def __init__(self, gct_info, i, mode_in):
        # gct_in is the string label
        self.representative = gct_info.gcts_labels[i]
        self.representative_np = str2np( self.representative )
        self.representative_vl = gct_info.gct_vl[i]
        # we might need representative rpcp
        self.members = [ self.representative ]
        # we might need rpcps for each member in a group for checking similarity
        self.members_np = [ str2np( self.representative ) ]
        self.members_rpcp = [ gct2relpcp( str2np( self.representative ) ) ]
        self.member_occurances = [ gct_info.gcts_occurances[i] ]
        self.members_vl = [ self.representative_vl ]
        # mode is an np array already
        self.mode = mode_in
        # occurances and probability to be defined after the group has formed
        self.occurances = []
        self.probability = []
    # end constructor
    def add_gct(self, gct_in, occurances_in, vl_in):
        ''' add the group and check if it needs to become representative '''
        if gct_in in self.members:
            print('GCT: ', gct_in, ' already in group!')
        else:
            self.members.append( gct_in )
            self.members_np.append( str2np( gct_in ) )
            self.members_rpcp.append( gct2relpcp( str2np( gct_in ) ) )
            self.member_occurances.append( occurances_in )
            self.members_vl.append( vl_in )
            if occurances_in >= max(self.member_occurances):
                self.representative = gct_in
                self.representative_np = str2np( gct_in )
                self.representative_vl = vl_in
    # end add_gct
    def check_membership(self, gct_in):
        ''' if diatonic sub or super set it's added in the group '''
        membership = False
        if self.is_super_subset(gct_in):
            membership = True
        return membership
    # end check_membership
    def is_super_subset(self, gct_in):
        # get np representation of the gct string
        gct_np = str2np( gct_in )
        # perform the following checks for each member of the group
        for group_member in self.members_np:
            response = True
            # 1) same root
            if gct_np[0] != group_member[0]:
                response = False
            # 2) check subset relations
            repr_rpcp = gct2relpcp( group_member )
            gct_rpcp = gct2relpcp( gct_np )
            # keep the sum of the LOGICAL_AND operator on the rpcps
            and_sum = np.sum( np.logical_and( repr_rpcp, gct_rpcp ) )
            if and_sum != np.sum( repr_rpcp ) and and_sum != np.sum( gct_rpcp ):
                response = False
            # 3) check of both have similar diatonic behavior
            tonality_rpcp = mode2relpcp( self.mode )
            chord_1_and_sum = np.sum( np.logical_and(tonality_rpcp, repr_rpcp) )
            chord_2_and_sum = np.sum( np.logical_and(tonality_rpcp, gct_rpcp) )
            is_diatonic_1 = chord_1_and_sum == np.sum(repr_rpcp)
            is_diatonic_2 = chord_2_and_sum == np.sum(gct_rpcp)
            if is_diatonic_1 != is_diatonic_2:
                response = False
            # if at least one member does the job, it's enough - quit checking
            if response:
                break;
        return response
    # end is_super_subset
# end GCT_group class

def make_initial_probabilities( m, f ):
    m.gct_group_info.gcts_initial_probabilities = np.zeros( len(m.gct_group_info.gcts_labels) )
    for i in range( len( f ) ):
        g = f[i]
        for l in g.members:
            m.gct_group_info.gcts_initial_probabilities[ i ] += m.gct_info.gcts_initial_probabilities[ m.gct_info.gcts_labels.index( l ) ]
    return m
# end make_initial_probabilities

def make_markov_matrix( m, f ):
    m.gct_group_info.gcts_markov = []
    # new transmat with one row for each representative
    delTr = np.zeros( ( len(m.gct_group_info.gcts_labels), len(m.gct_group_info.gcts_labels) ) );
    tmpMat = np.zeros( ( len(m.gct_group_info.gcts_labels), len(m.gct_info.gcts_labels) ) );
    
    # for each representative, add the member transition probabilities
    for i in range( len(f) ):
        g = f[i]
        tmpSum = 0
        for j in range( len(g.members) ):
            l = g.members[j]
            # find the idx of member in the list of all gct labels
            idx = m.gct_info.gcts_labels.index( l )
            # get the probability of this idx - to weigh the contribution
            tmp_pr = m.gct_info.gcts_probabilities[idx]
            # add the row of the idx to the sum
            tmpMat[i,:] = tmpMat[i,:] + tmp_pr*m.gct_info.gcts_markov[idx, :];
            tmpSum += tmp_pr
        tmpMat[i,:] = tmpMat[i,:]/tmpSum;
    # squize columns
    for i in range( len(f) ):
        g = f[i]
        tmpColSum = np.zeros( len(f) )
        for j in range( len(g.members) ):
            l = g.members[j]
            # find the idx of member in the list of all gct labels
            idx = m.gct_info.gcts_labels.index( l )
            tmpColSum += tmpMat[:, idx]
        delTr[:,i] = tmpColSum
    # neutralise diagonal?
    m.gct_group_info.gcts_markov = delTr
    return m
# end make_markov_matrix

def make_info_structure( m, f ):
    # self.gcts_array = [] # GCTgroups don't have it
    # self.gcts_counter = [] # GCTgroups don't have it
    # self.gct_group_structures = [] # only GCTgroups have it
    m.gct_group_info.gct_group_structures = f
    # self.gcts_membership_dictionary = {} # only GCTgroups have it
    gctg_dict = {}
    m.gct_group_info.gcts_labels = []
    m.gct_group_info.gcts_relative_pcs = []
    m.gct_group_info.gcts_occurances = []
    m.gct_group_info.gcts_probabilities = []
    for g in f:
        gctg_dict[g.representative] = g
        # self.gcts_labels = []
        m.gct_group_info.gcts_labels.append( g.representative )
        # self.gcts_relative_pcs = []
        # in groups this should be a list of lists for the rpcp of each group
        tmp_rpcps_list = []
        tmp_member_probs_list = []
        for i in range( len( g.members_np ) ):
            mmbr_np = g.members_np[ i ]
            tmp_rpcps_list.append( gct2relpcp( mmbr_np ) )
            if sum(g.member_occurances) > 0:
                tmp_member_probs_list.append( g.member_occurances[i]/sum(g.member_occurances) )
            else:
                tmp_member_probs_list.append( 0.0 )
        m.gct_group_info.gcts_relative_pcs.append( tmp_rpcps_list )
        # self.gcts_occurances = [] # __NEW__
        m.gct_group_info.gcts_occurances.append( g.occurances )
        # self.gcts_probabilities = [] # __NEW__
        m.gct_group_info.gcts_probabilities.append( tmp_member_probs_list )
    m.gct_group_info.gcts_membership_dictionary = gctg_dict
    m.gct_group_info.gct_vl_dict = m.gct_info.gct_vl_dict
    # self.gcts_initial_array = []
    # m.gct_group_info.gcts_initial_array = [] # GCTgroups don't need it
    # self.gcts_initial_counter = [] # GCTgroups don't have it
    # self.gcts_initial_probabilities = []
    m = make_initial_probabilities(m, f)
    # self.gcts_transitions_sum = []
    # m.gct_group_info.gcts_transitions_sum = [] # GCTgroups don't need it
    # self.gcts_markov = []
    m = make_markov_matrix(m, f)
    return m
# end make_info_structure

def group_gcts_of_mode(m):
    gct_groups = []
    to_not_include = []
    for i in range( len( m.gct_info.gcts_labels ) ):
        if i == 0:
            gct_groups.append( GCT_group( m.gct_info, i, m.mode ) )
        else:
            b = False
            # keep the index of the group that the member entered
            tmp_group_idx = 0
            for j in range( len(gct_groups) ):
                g = gct_groups[j]
                # check if membership applies - if so, gct is already added
                b = g.check_membership( m.gct_info.gcts_labels[i] )
                if b:
                    g.add_gct( m.gct_info.gcts_labels[i], m.gct_info.gcts_occurances[i], m.gct_info.gct_vl[i] )
                    tmp_group_idx = j
                    break;
            if not b:
                gct_groups.append( GCT_group( m.gct_info, i, m.mode ) )
            else:
                # if a member is found, it might be used to combine other groups
                # doing a second check
                # keep group indexes that need to be NOT included in the final group list
                for j in range( len(gct_groups) ):
                    # should not look at the group of the found member
                    if j != tmp_group_idx:
                        g = gct_groups[j]
                        b = g.check_membership( m.gct_info.gcts_labels[i] )
                        # if it does relate to another group
                        if b:
                            # this group needs to NOT be included in the final list
                            to_not_include.append( j )
                            # add all members of the other group here
                            for k in range( len( g.members ) ):
                                gct_groups[tmp_group_idx].add_gct( g.members[k], g.member_occurances[k], g.members_vl[k] )
    # form the final list of groups
    final_groups_list = []
    all_group_occurances = 0
    for i in range( len( gct_groups ) ):
        if i not in to_not_include:
            # compute group occurances
            gct_groups[i].occurances = sum( gct_groups[i].member_occurances )
            # keep the sum for computing probabilities
            all_group_occurances += gct_groups[i].occurances
            final_groups_list.append( gct_groups[i] )
    # compute probabilities
    if all_group_occurances > 0:
        for g in final_groups_list:
            g.probability = g.occurances/all_group_occurances
    # NEED TO CONSTRUCT gct_group_info TO ANOTHER FUNCTION
    m = make_info_structure( m, final_groups_list )
    # m.gct_group_info = gct_group_info
    # NOT SURE: this function should return and construct two things: gct_groups list and gct_membership dictionary
    return m

def group_gcts_of_idiom(idiom):
    for m in idiom.modes.values():
        m = group_gcts_of_mode(m)
    return idiom
# end group_gcts_of_idiom