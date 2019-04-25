#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:30:17 2019

@author: maximoskaliakatsos-papakostas
"""

'''

Functions for printing information of an idiom.

HOW TO RUN:
    
print_idiom(idiom_name, output_file='', mode='all', include='all')

For more information SEE FUNCTION print_idiom at the end of the file

EXAMPLES:
# first import module
import idiom_printer as prnt
# ex1: printing all idiom info on screen
prnt.print_idiom('BachChorales')
# ex2: printing all info in files
prnt.print_idiom('BachChorales', output_file='BC_info')
# ex3: printing single mode info for gcts and markov in files
prnt.print_idiom('BachChorales', output_file='BC_major_info', mode='[0, 2, 4, 5, 7, 9, 11]', include=['gcts', 'markov'])

INPUTS:
        
    - idiom_name: necessary input, the file name of the saved idiom, has to be
    the same name found in the 'trained_idioms' folder
    
    - output_file: if empty, results will not be saved to files but only printed
    on screen; if not empty, the file name should NOT have extension and multiple
    file will be generated for lists, matrices and figures
    
    - mode: if 'all', information for all modes will be printed (either on 
    screen or in files), else, the exact string of the mode array should be
    given, e.g. '[0 2 4 5 7 9 11]' for major etc. If not 'all', only
    A SINGLE mode is accepted as argument
    
    - include: if 'all', all information will be printed (either on screen or
    in files); else, include can be:
        - 'gcts'
        - 'markov'
        - 'cadences'
        - 'gct_families'
        - 'markov_families'
        - 'bbvl'

'''

import pickle
import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd + '/CM_train')
sys.path.insert(0, cwd + '/CM_auxiliary')
import CM_TR_TrainingIdiom_class as tic
import CM_TonalityGrouping_classes as tgc
import numpy as np
import matplotlib.pyplot as plt

def print_gcts(m, output_file):
    # get necessary elements
    g = m.gct_info
    # if no output file is requested
    if output_file == '':
        print('GCTs =================================================')
        for i in range( len( g.gcts_labels ) ):
            # gcts_labels
            # gcts_occurances
            # gcts_probabilities
            # gcts_relative_pcs
            print( "{0:20} \t {1}".format(g.gcts_labels[i] + '\t', str(g.gcts_occurances[i])) + '\t' + "{:.4f}".format(g.gcts_probabilities[i]) + '\t' + "{:.4f}".format(g.gcts_initial_probabilities[i]) + '\t' + str(g.gcts_relative_pcs[i]) )
    else:
        with open(output_file+'.txt', 'a') as the_file:
            the_file.write('GCTs =================================================' + '\n')
            the_file.write( "{0:20} \t Occ. \t Probs \t Init. \t Relative PCs ".format('Labels') + '\n')
            for i in range( len( g.gcts_labels ) ):
                the_file.write( "{0:20} \t {1}".format(g.gcts_labels[i], str(g.gcts_occurances[i])) + '\t' + "{:.4f}".format(g.gcts_probabilities[i]) + '\t' + "{:.4f}".format(g.gcts_initial_probabilities[i]) + '\t' + str(g.gcts_relative_pcs[i]) + '\n' )
# end print_gcts
def print_gct_families(m, output_file):
    # get necessary elements
    g = m.gct_group_info
    # if no output file is requested
    if output_file == '':
        print('GCT families ==========================================')
        for i in range( len( g.gcts_labels ) ):
            # gcts_labels
            # gcts_occurances
            # gcts_probabilities
            # gcts_relative_pcs
            print( "{0:20} \t {1}".format(g.gcts_labels[i] + '\t', str(g.gcts_occurances[i])) + '\n' )
    else:
        with open(output_file+'.txt', 'a') as the_file:
            the_file.write('GCT families ======================================' + '\n')
            the_file.write( "{0:20} \t Occ. ".format('Labels') + '\n')
            for i in range( len( g.gcts_labels ) ):
                the_file.write( "{0:20} \t {1}".format(g.gcts_labels[i], str(g.gcts_occurances[i])) + '\n' )
            # print membership dictionary
            the_file.write( 'Membership:' +'\n' )
            memb_dict = g.gcts_membership_dictionary
            memb_keys = list( memb_dict.keys() )
            for i in range( len( memb_keys ) ):
                the_file.write( "{0:20}:".format( memb_keys[i] ) )
                members = memb_dict[ memb_keys[i] ].members
                for j in range( len(members) ):
                    # skip first - its the representative printed above
                    if j > 0:
                        the_file.write( members[j] + '\t' )
                the_file.write( '\n' )
# end print_gct_families
def print_cadences(m, output_file):
    # types of cadences
    cad_type = ['final', 'intermediate']
    for ct in range( len( cad_type ) ):
        # get necessary elements
        c = m.cadences[cad_type[ct]]
        # if no output file is requested
        if output_file == '':
            print('Cadences ' + cad_type[ct] + ' ==========================================')
            for i in range( len( c.cadence_labels ) ):
                # gcts_labels
                # gcts_occurances
                # gcts_probabilities
                # gcts_relative_pcs
                print( "{0:20} \t {1}".format(c.cadence_labels[i] + '\t', str(c.cadence_occurances[i])) + '\t' + "{:.4f}".format(c.cadence_probabilities[i]) + '\n' )
        else:
            with open(output_file+'.txt', 'a') as the_file:
                the_file.write('Cadences ' + cad_type[ct] + ' ==========================================' + '\n')
                the_file.write( "{0:30} \t Occ. \t Probs".format('Labels') + '\n')
                for i in range( len( c.cadence_labels ) ):
                    the_file.write( "{0:20} \t {1}".format(c.cadence_labels[i] + '\t', str(c.cadence_occurances[i])) + '\t' + "{:.4f}".format(c.cadence_probabilities[i]) + '\n' )
# end print_cadences
def print_markov(m, output_file):
    # get necessary elements
    g = m.gct_info
    tmpfig = plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(g.gcts_markov, cmap='gray_r', interpolation='none');
    plt.xticks(range(len(g.gcts_labels)), g.gcts_labels, rotation='vertical')
    plt.yticks(range(len(g.gcts_labels)), g.gcts_labels)
    tmpfig.savefig(output_file + '.png', format='png', dpi=300, bbox_inches="tight")
    plt.clf()
# end print_markov
def print_markov_families(m, output_file):
    # get necessary elements
    g = m.gct_group_info
    tmpfig = plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(g.gcts_markov, cmap='gray_r', interpolation='none');
    plt.xticks(range(len(g.gcts_labels)), g.gcts_labels, rotation='vertical')
    plt.yticks(range(len(g.gcts_labels)), g.gcts_labels)
    tmpfig.savefig(output_file + '_families.png', format='png', dpi=300, bbox_inches="tight")
    plt.clf()
# end print_markov_families

def print_element(m, output_file, el):
    # if no output file is requested
    if output_file == '':
        print('Printing: ', el)
        if el == 'gcts':
            print_gcts(m, output_file)
        elif el == 'gct_families':
            print_gct_families(m, output_file)
        elif el == 'markov':
            print('markov')
        elif el == 'cadences':
            print_cadences(m, output_file)
        elif el == 'markov_families':
            print('markov_families')
        elif el == 'bbvl':
            print('bbvl')
        else:
            print('this is NOT supposed to be printed!')
    else:
        with open(output_file+'.txt', 'a') as the_file:
            the_file.write('Printing: '+el+'\n' )
        print('Printing: ', el)
        if el == 'gcts':
            print_gcts(m, output_file)
        elif el == 'gct_families':
            print_gct_families(m, output_file)
        elif el == 'markov':
            print_markov(m, output_file)
        elif el == 'cadences':
            print_cadences(m, output_file)
        elif el == 'markov_families':
            print_markov_families(m, output_file)
        elif el == 'bbvl':
            print('bbvl')
        else:
            print('this is NOT supposed to be printed!')
# end print_element

def print_mode( m, output_file, include ):
    print('printing mode: ', m.mode_name)
    print('printing info for: ', include)
    print('in output file: ', output_file)
    elements = ['gcts', 'markov', 'cadences', 'gct_families', 'markov_families']
    # elements = ['gcts', 'markov', 'cadences', 'gct_families', 'markov_families', 'bbvl']
    # check if file has not been requested
    if output_file == '':
        print('Mode: ', m.mode_name)
    else:
        # initialise file - make a clean file so other functions can append
        with open(output_file+'.txt', 'a') as the_file:
            the_file.write('Mode: '+m.mode_name+'\n')
    # check which elements have been requested
    if include == 'all':
        # print all elements for mode
        for el in elements:
            print_element( m, output_file, el)
    else:
        for el in include:
            # check if user given element not in list
            if el not in elements:
                print('Element: ', el, ' not in understood.')
                print('Currently understanding the following elements: ', elements)
            else:
                print_element( m, output_file, el)
# end print_mode

def print_idiom(idiom_name, output_file='', mode='all', include='all'):
    '''
    INPUTS:
        
    - idiom_name: necessary input, the file name of the saved idiom, has to be
    the same name found in the 'trained_idioms' folder
    
    - output_file: if empty, results will not be saved to files but only printed
    on screen; if not empty, the file name should NOT have extension and multiple
    file will be generated for lists, matrices and figures
    
    - mode: if 'all', information for all modes will be printed (either on 
    screen or in files), else, the exact string of the mode array should be
    given, e.g. '[0 2 4 5 7 9 11]' for major etc. If not 'all', only
    A SINGLE mode is accepted as argument
    
    - include: if 'all', all information will be printed (either on screen or
    in files); else, include can be:
        - 'gcts'
        - 'markov'
        - 'cadences'
        - 'gct_families'
        - 'markov_families'
        - 'bbvl'
    
    OUTPUTS:
    - NO OUTPUT
    '''
    
    # load idiom
    with open(cwd+'/trained_idioms/'+idiom_name+'.pickle', 'rb') as handle:
        idiom = pickle.load(handle)
    # check if file has not been requested
    if output_file == '':
        print('Printing begins: ')
    else:
        # make a folder to put every produced file in
        folder_name = cwd+os.sep+'IDIOM_LOG'+os.sep+output_file+os.sep
        if not os.path.exists(folder_name):
            os.makedirs( folder_name )
        # append path to output file name
        output_file = folder_name+output_file
        # initialise file - make a clean file so other functions can append
        with open(output_file+'.txt', 'w') as the_file:
            the_file.write('File initialisation\n')
    # get all modes of idiom
    modes = list( idiom.modes.keys() )
    # check if specific mode needs to be printed
    if mode == 'all':
        print('DEBUG - printing all modes')
        # print for all modes
        for m_id in modes:
            # get mode
            m = idiom.modes[ m_id ]
            # print information from mode
            print_mode( m, output_file, include )
    else:
        # if not all modes are give, check if given mode in list
        if mode in modes:
            print('DEBUG - printing selected mode')
            m = idiom.modes[ mode ]
            print_mode( m, output_file, include )
        else:
            print('error: given mode not in list')
            print('given mode is: ', mode)
            print('available modes for selected idiom are:', modes)
    return idiom
# end print_idiom