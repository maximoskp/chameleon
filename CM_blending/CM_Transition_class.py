#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:16:46 2018

@author: maximoskaliakatsos-papakostas
"""

import numpy as np
import copy
import computeDIC as dic
import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_Misc_Aux_functions as maf

class Transition:
    def __init__(self, c1_np, c2_np):
        # print('initialising transition with chords: ', c1_np, ' - ', c2_np)
        self.properties_list = ['from_chord_np', 'to_chord_np', 'from_rpc', 'to_rpc', 'dic_has_0', 'dic_has_1', 'dic_has_N1', 'asc_semitone_to_root', 'desc_semitone_to_root', 'semitone_to_root']
        # remember to retrive object property from string of attribute as:
        # attr = getattr( obj, STR_ATTR )
        self.from_chord_np = {
            'property': c1_np,
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'chord_name',
            'necessary': True,
        }
        self.to_chord_np = {
            'property': c2_np,
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'chord_name',
            'necessary': True
        }
        self.from_rpc = {
            'property': np.mod( c1_np[0]+c1_np[1:], 12 ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'subset_match',
            'necessary': True
        }
        self.to_rpc = {
            'property': np.mod( c2_np[0]+c2_np[1:], 12 ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'subset_match',
            'necessary': True,
        }
        self.dic_has_0 = {
            'property': self.compute_dic_value( c1_np, c2_np, 0 ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.dic_has_1 = {
            'property': self.compute_dic_value( c1_np, c2_np, 1 ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.dic_has_N1 = {
            'property': self.compute_dic_value( c1_np, c2_np, -1 ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.asc_semitone_to_root = {
            'property': (11 in c1_np) and (0 in c2_np),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.desc_semitone_to_root = {
            'property': (1 in c1_np) and (0 in c2_np),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.semitone_to_root = {
            'property': ( (1 in c1_np) and (0 in c2_np) ) or ( (11 in c1_np) and (0 in c2_np) ),
            'priority_idiom': 0,
            'priority_other': 0,
            'priority': 0,
            'matching': 'boolean',
            'necessary': False,
        }
        self.blending_score = 0
    # end constructor

    def compute_dic_value(self, c1, c2, d):
        b = False
        p1 = np.mod( c1[0]+c1[1:], 12 )
        p2 = np.mod( c2[0]+c2[1:], 12 )
        v,ids = dic.computeDICfromMIDI(p1,p2)
        if v[ np.where( ids == d )[0][0] ] > 0:
            b = True
        return b
    # end compute_dic_value
# end Transition class

def check_generic_constraints(t, g):
    # assume it satisfies generic space constraints
    b = True
    for p_name in t.properties_list:
        # get generic space property
        pg = getattr( g, p_name )
        # check if generic space has a restriction for this property
        if pg['property'] is not 'empty':
            # get transition property
            pt = getattr( t, p_name )
            if pt['matching'] == 'chord_name':
                if not np.array_equal( pt['property'] , pg['property'] ):
                    b = False
                    break
            elif pt['matching'] == 'boolean':
                if pt['property'] != pg['property']:
                    b = False
                    break
            elif pt['matching'] == 'subset_match':
                if not np.all( np.isin( pg['property'] , pt['property'] ) ):
                    b = False
                    break
    return b;
# end check_generic_constraints
def rate_blend(b, i1, i2):
    # initiase a score of zero
    s = 0
    for p_name in b.properties_list:
        # get blend property
        pb = getattr( b, p_name )
        # get properties of inputs
        pi1 = getattr( i1, p_name )
        pi2 = getattr( i2, p_name )
        if pb['matching'] == 'chord_name':
            if np.array_equal( pb['property'] , pi1['property'] ):
                # get half reward from each input
                s = s + pi1['priority']/2
            if np.array_equal( pb['property'] , pi2['property'] ):
                # get half reward from each input
                s = s + pi2['priority']/2
        elif pb['matching'] == 'boolean':
            if pb['property'] == pi1['property']:
                s = s + pi1['priority']/2
            if pb['property'] != pi2['property']:
                s = s + pi2['priority']/2
        elif pb['matching'] == 'subset_match':
            inclusion = np.isin( pi1['property'] , pb['property'] )
            if np.any( inclusion ):
                s = s + np.sum( pi1['priority'][inclusion] )/(2*np.sum(inclusion))
            inclusion = np.isin( pi2['property'] , pb['property'] )
            if np.any( inclusion ):
                s = s + np.sum( pi2['priority'][inclusion] )/(2*np.sum(inclusion))
    b.blending_score = s
    return b
# end rate_blend

def compute_priorities(trans, intra_trans, other_trans):
    # computes priorities for transition ontologies
    # trans: transitions to compute priorities for
    # idiom_trans: transition ontologies of the idiom that the trans belongs to
    # idiom_other: transition ontologies of the idiom that the trans doesn't belong to
    # HOME =================================================================
    # for each property, check how often it is used in the idioms
    for p_name in trans.properties_list:
        p = getattr( trans, p_name )
        # in case property is considered a single element
        if p['matching'] == 'chord_name' or p['matching'] == 'boolean':
            intra_idiom_count = 0
            for tr in intra_trans:
                t = getattr( tr, p_name )
                if p['matching'] == 'chord_name':
                    if np.array_equal( p['property'], t['property'] ):
                        intra_idiom_count = intra_idiom_count + 1
                elif p['matching'] == 'boolean':
                    if p['property'] == t['property']:
                        intra_idiom_count = intra_idiom_count + 1
            # assign property value
            p['priority_idiom'] = intra_idiom_count/len( intra_trans )
        # in case property is an array of properties
        elif p['matching'] == 'subset_match':
            sub_intra_count = np.zeros( len(p['property']) )
            for i in range( len( p['property'] ) ):
                for tr in intra_trans:
                    t = getattr( tr, p_name )
                    if p['property'][i] in t['property']:
                        sub_intra_count[i] = sub_intra_count[i] + 1
            # assign property value
            p['priority_idiom'] = sub_intra_count/len( intra_trans )
        else:
            print('Unknown matching type!')
    # AWAY =================================================================
    # for each property, check how often it is used in the idioms
    for p_name in trans.properties_list:
        p = getattr( trans, p_name )
        # in case property is considered a single element
        if p['matching'] == 'chord_name' or p['matching'] == 'boolean':
            other_idiom_count = 0
            for tr in other_trans:
                t = getattr( tr, p_name )
                if p['matching'] == 'chord_name':
                    if np.array_equal( p['property'], t['property'] ):
                        other_idiom_count = other_idiom_count + 1
                elif p['matching'] == 'boolean':
                    if p['property'] == t['property']:
                        other_idiom_count = other_idiom_count + 1
            # assign property value
            p['priority_other'] = 1 - other_idiom_count/len( other_trans )
        # in case property is an array of properties
        elif p['matching'] == 'subset_match':
            sub_other_count = np.zeros( len(p['property']) )
            for i in range( len( p['property'] ) ):
                for tr in other_trans:
                    t = getattr( tr, p_name )
                    if p['property'][i] in t['property']:
                        sub_other_count[i] = sub_other_count[i] + 1
            # assign property value
            p['priority_other'] = 1 - sub_other_count/len( other_trans )
        else:
            print('Unknown matching type!')
        p['priority'] = 0.5*p['priority_idiom'] + 0.5*p['priority_other']
        # print(p['priority'])
    return trans

def compute_generic_space(t1, t2):
    g = Transition( t1.from_chord_np['property'] , t1.to_chord_np['property'])
    for p_name in t1.properties_list:
        p1 = getattr( t1, p_name )
        p2 = getattr( t2, p_name )
        pg = getattr( g, p_name )
        if p1['matching'] == 'chord_name':
            if not np.array_equal( p1['property'], p2['property'] ):
                pg['property'] = 'empty'
        elif p1['matching'] == 'boolean':
            if p1['property'] != p2['property']:
                pg['property'] = 'empty'
        elif p1['matching'] == 'subset_match':
            pg['property'] = np.intersect1d( p1['property'], p2['property'] )
            if len( pg['property'] ) == 0:
                pg['property'] = 'empty'
    return g