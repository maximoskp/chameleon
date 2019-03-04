#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:02:22 2018

@author: maximoskaliakatsos-papakostas
"""

from music21 import *
import numpy as np
import os
import glob
import copy

# the user should give the file name - the folder with the files
def get_parts_np_from_file(fileName, parts_for_surface, time_res):
    # INPUTS:
    # fileName - string: the name of the file - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrix in np array
    # m_length: the number of columns in the matrix
    
    p = converter.parse(fileName)
    
    # tmp list that will include the lists of all pitches
    tmp_all_pitches = []
    
    if parts_for_surface == 'all':
        parts_for_surface = range(len(p.parts))
    
    # find the length of the largest part for concatenating midi
    # this is used if measures are empty and we have to go with parts
    largest_part_length = 0
    for ii in parts_for_surface:
        tmp_part = p.parts[ii]
        notes = tmp_part.flat.notes
        if largest_part_length < eval(str(notes[-1].offset)):
            largest_part_length = eval(str(notes[-1].offset))
    # make surface matrix
    for i in parts_for_surface:
        # get part
        tmp_part = p.parts[i]
        # tmp list that will include the lists of all pitches in the part
        tmp_part_pitches = []
        # get array of measures
        measures = [m for m in tmp_part.getElementsByClass('Measure')]
        if measures:
            # for all measures
            for m in measures:
                # get time signature of measure
                ts = m.flat.timeSignature
                if ts != None:
                    # tmp_ts = ts[0]
                    if ts.numerator == 4 and ts.denominator == 4:
                        measureLength = time_res
                    elif ts.numerator == 3 and ts.denominator == 4:
                        measureLength = int(3.0*time_res/4.0)
                    elif ts.numerator == 3 and ts.denominator == 8:
                        measureLength = int(3.0*time_res/8.0)
                    elif ts.numerator == 2 and ts.denominator == 4:
                        measureLength = int(time_res/2.0)
                    elif ts.numerator == 12 and ts.denominator == 8:
                        measureLength = int(3.0*time_res/2.0)
                    elif ts.numerator == 3 and ts.denominator == 2:
                        measureLength = int(3.0*time_res/2.0)
                    else:
                        print("unknown time signature: ", ts.numerator, ts.denominator)
                notes = m.flat.notes
                # tmp list that stores the pitches for the measure
                tmp_measure_pitches = np.zeros((128, measureLength))
                for n in notes:
                    offset_value = int( eval(str(n.offset))*time_res/4.0 )
                    duration_value = int(n.duration.quarterLength*time_res/4.0)
                    if n.isChord:
                        for nn in n.pitches:
                            midi_number = nn.midi
                            # print(midi_number, " - ", offset_value, " - ", duration_value)
                            tmp_measure_pitches[midi_number, offset_value] = duration_value
                    else:
                        midi_number = n.pitch.midi
                        tmp_measure_pitches[midi_number, offset_value] = duration_value
                
                # tmp_all_pitches.append(tmp_measure_pitches)
                if len(tmp_part_pitches) == 0:
                    tmp_part_pitches = np.array(tmp_measure_pitches)
                else:
                    tmp_part_pitches = np.hstack((tmp_part_pitches, tmp_measure_pitches))
        else:
            # do the same for part
            notes = tmp_part.flat.notes
            # get part length
            part_length = largest_part_length
            # tmp list that stores the pitches for the measure
            tmp_part_pitches = np.zeros((128, int( part_length*time_res/4.0 )+1 ))
            for n in notes:
                offset_value = int( eval(str(n.offset))*time_res/4.0 )
                duration_value = int(n.duration.quarterLength*time_res/4.0)
                if n.isChord:
                    for nn in n.pitches:
                        midi_number = nn.midi
                        # print(midi_number, " - ", offset_value, " - ", duration_value)
                        tmp_part_pitches[midi_number, offset_value] = duration_value
                else:
                    midi_number = n.pitch.midi
                    tmp_part_pitches[midi_number, offset_value] = duration_value
        # end if measures
        tmp_all_pitches.append(tmp_part_pitches)
    # end for parts
    all_pitches = np.array(tmp_all_pitches[0])
    for a in tmp_all_pitches:
        # print('tmp_all_pitches: ', tmp_all_pitches)
        # print('all_pitches: ', all_pitches)
        all_pitches[ all_pitches == 0 ] = a[ all_pitches == 0 ]
        
    return all_pitches, np.size(all_pitches, axis=1)

def get_rel_pcp_np_from_file(fileName, parts_for_surface, time_res):
    # INPUTS:
    # fileName - string: the name of the file - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrix in np array
    # m_length: the number of columns in the matrix
    
    p = converter.parse(fileName)
    
    # tmp list that will include the lists of all pitches
    tmp_all_pitches = []
    
    if parts_for_surface == 'all':
        parts_for_surface = range(len(p.parts))
    
    # make surface matrix
    for i in parts_for_surface:
        # get part
        tmp_part = p.parts[i]
        # tmp list that will include the lists of all pitches in the part
        tmp_part_pitches = []
        # get array of measures
        measures = [m for m in tmp_part.getElementsByClass('Measure')]
        # for all measures
        for m in measures:
            # get time signature of measure
            ts = m.flat.timeSignature
            if ts != None:
                # tmp_ts = ts[0]
                if ts.numerator == 4 and ts.denominator == 4:
                    measureLength = time_res
                elif ts.numerator == 3 and ts.denominator == 4:
                    measureLength = int(3.0*time_res/4.0)
                elif ts.numerator == 3 and ts.denominator == 8:
                    measureLength = int(3.0*time_res/8.0)
                else:
                    print("unknown time signature: ", ts.numerator, ts.denominator)
            notes = m.flat.notes
            # tmp list that stores the pitches for the measure
            tmp_measure_pitches = np.zeros((128, measureLength))
            for n in notes:
                offset_value = int( eval(str(n.offset))*time_res/4.0 )
                duration_value = int(n.duration.quarterLength*time_res/4.0)
                if n.isChord:
                    for nn in n.pitches:
                        midi_number = nn.midi
                        # print(midi_number, " - ", offset_value, " - ", duration_value)
                        tmp_measure_pitches[midi_number, offset_value] = duration_value
                else:
                    midi_number = n.pitch.midi
                    tmp_measure_pitches[midi_number, offset_value] = duration_value
            
            # tmp_all_pitches.append(tmp_measure_pitches)
            if len(tmp_part_pitches) == 0:
                tmp_part_pitches = np.array(tmp_measure_pitches)
            else:
                tmp_part_pitches = np.hstack((tmp_part_pitches, tmp_measure_pitches))
        
        tmp_all_pitches.append(tmp_part_pitches)
        
        all_pitches = np.array(tmp_all_pitches[0])
        for a in tmp_all_pitches:
            all_pitches[ all_pitches == 0 ] = a[ all_pitches == 0 ]
        
    return all_pitches, np.size(all_pitches, axis=1)
# end

# running for all files in folder
def get_parts_3Dnp_from_folder(folderName, parts_for_surface, time_res):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrices in np array (n_pieces, 128, max_len)
    # m_length: the number of columns in each matrix (n_pieces, )
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    for i in range(len(all_matrices)):
        m = all_matrices[i]
        # check if padding needed
        if m.shape[1] < max_length:
            # make a padding matrix
            padder = -1.0*np.ones((128, max_length-m.shape[1]))
            all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.array(all_matrices)
    return all_matrices, all_lengths

def compute_features(m):
    horizontal_range = 4
    lowest_limit = np.min( np.where( np.sum(m, axis=1) ) )
    highest_limit = np.max( np.where( np.sum(m, axis=1) ) )
    prev_lowest = lowest_limit
    prev_highest = highest_limit
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, m.shape[1]) )
    # regularise register
    f[1,:] = 0.5*np.ones( (1, m.shape[1]) )
    for i in range(horizontal_range, m.shape[1]-horizontal_range):
        tmp_1 = np.sum( m[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1
        if len(tmp_1) == 0:
            tmp_1 = [0]
        f[0,i] = np.mean( tmp_1 )
        # smoothening
        # f[0,i] = np.mean(f[0, (i-horizontal_range):(i+horizontal_range)])
        
        '''
        POLYPHONY feature
        tmpMat = m[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        tmp_2 = np.sum(tmpMat_part, axis=0)/np.max( np.sum(m, axis=0) )
        if len(tmp_2) == 0:
            tmp_2 = [0]
        f[1,i] = np.mean( tmp_2 )
        '''
        # smoothening
        # f[1,i] = np.mean(f[1, (i-horizontal_range):(i+horizontal_range)])
        
        # register feature
        tmp_sum = np.sum(m[:, (i-horizontal_range):(i+horizontal_range) ], axis=1)
        if np.sum(tmp_sum) > 0:
            lowest_note = np.min( np.where( tmp_sum ) )
            highest_note = np.max( np.where( tmp_sum ) )
            prev_lowest = lowest_note
            prev_highest = highest_note
        else:
            lowest_note = prev_lowest
            highest_note = prev_highest
        f[1,i] = ((lowest_note + highest_note)/2.0 - lowest_limit)/(highest_limit - lowest_limit)
        
    # moving average
    f[0,:] = np.convolve(f[0,:], np.ones(horizontal_range)/horizontal_range, mode="same")
    f[1,:] = np.convolve(f[1,:], np.ones(horizontal_range)/horizontal_range, mode="same")
    # swapping
    # f[[0,1]] = f[[1,0]]
    return f
# end compute_features

def compute_pcp( c ):
    pcp = np.zeros( 12 )
    for i in range( len(c) ):
        if c[i] > 0:
            pcp[ int(i%12) ] = 1
    return pcp
# end compute_pcp

def compute_vertical_tension( c ):
    cof_dists = np.array( [5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5] )
    notes_num = np.count_nonzero( c )
    total_dist = 0
    for i in range( len(c) ):
        for j in range( len(c) ):
            if i != j and c[i] > 0 and c[j] > 0:
                d = np.abs( i-j )
                d_cof = cof_dists[ d-1 ]
                total_dist += d_cof
    if notes_num > 1:
        total_dist /= 2.0*notes_num
    return total_dist
# end compute_vertical_tension

def compute_horizontal_tension( p, c ):
    cof_dists = np.array( [5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5] )
    notes_num = np.count_nonzero( p ) + np.count_nonzero( c )
    total_dist = 0
    for i in range( len(p) ):
        for j in range( len(c) ):
            if p[i] > 0 and c[j] > 0:
                d = np.abs( i-j )
                d_cof = cof_dists[ d-1 ]
                total_dist += d_cof
    if notes_num > 1:
        total_dist /= notes_num
    return total_dist
# end compute_horizontal_tension

def compute_tension(m, normalize=False, smoothen=False, smoothingWindow=4):
    f = np.zeros( m.shape[1] )
    prev_pcp = np.zeros(12)
    horizontal_tension = 0
    vertical_tension = 0
    for i in range(m.shape[1]):
        curr_pcp = compute_pcp( m[:,i] )
        # print('curr_pcp: ', curr_pcp)
        if np.sum( curr_pcp ) > 0:
            vertical_tension = compute_vertical_tension( curr_pcp )
            horizontal_tension = compute_horizontal_tension( prev_pcp, curr_pcp )
        f[i] = vertical_tension + horizontal_tension
    if normalize:
        # normalise
        max_val = np.max( f )
        if max_val > 0:
            f /= max_val
    if smoothen:
        # moving average
        f = np.convolve(f, np.ones(4)/4, mode="same")
    return f
# end compute_features

# running for all files in folder but returns all pieces concatenated in a large matrix
def get_concat_parts_np_from_folder(folderName, parts_for_surface, time_res, transpose = False, bin_out=True, voice_aug=False, sparse_aug=False, register_aug=[], padding=False):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    if len(allDocs) is 0:
        print("Didn't find .xml file - looking for .mxl")
        allDocs = glob.glob(folderName + os.sep + "*.mxl")
    if len(allDocs) is 0:
        print("Didn't find .mxl file - looking for .mid")
        allDocs = glob.glob(folderName + os.sep + "*.mid")
        
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
        # octave augmentation if is binary
        if bin_out:
            new_mat = np.zeros( all_matrices.shape )
            for i in range(all_matrices.shape[1]):
                new_mat[:,i] = np.logical_or( np.logical_or( all_matrices[:,i], np.roll(all_matrices[:,i], 12) ), np.roll(all_matrices[:,i], -12) )
            all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.9/(1+ 0.2*(i%16) + 0.5*(i%8) + 0.5*(i%4) + 4*(i%2)):
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if register augmentation has been asked
    if len(register_aug) > 0:
        # do register augmentation
        print('performing register augmentation')
        new_all_matrices = np.array(all_matrices)
        for shift in register_aug:
            print('range shift: ', shift)
            new_mat = np.zeros( all_matrices.shape )
            for i in range(all_matrices.shape[1]):
                # print('i: ', i)
                if np.count_nonzero(all_matrices[:,i]) >= 1:
                    new_mat[:,i] = np.roll(all_matrices[:,i], shift)
            new_all_matrices = np.hstack( (new_all_matrices, new_mat) )
        all_matrices = np.array( new_all_matrices )
    
    # compute feature matrix
    print('computing feature matrix')
    f = compute_features(all_matrices)
    '''
    horizontal_range = 8
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, all_matrices.shape[1]) )
    for i in range(horizontal_range, all_matrices.shape[1]-horizontal_range):
        f[0,i] = np.mean( np.sum( all_matrices[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1 )
        tmpMat = all_matrices[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        f[1,i] = np.mean( np.sum(tmpMat_part, axis=0)/4.0 )
    '''
    
    return all_matrices, f

def get_concat_tension_np_from_folder(folderName, parts_for_surface, time_res, transpose = False, bin_out=True, voice_aug=False, sparse_aug=False, register_aug=[], padding=False):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    if len(allDocs) is 0:
        print("Didn't find .xml file - looking for .mxl")
        allDocs = glob.glob(folderName + os.sep + "*.mxl")
    if len(allDocs) is 0:
        print("Didn't find .mxl file - looking for .mid")
        allDocs = glob.glob(folderName + os.sep + "*.mid")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        print('Running for ', fileName)
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
        # octave augmentation if is binary
        if bin_out:
            new_mat = np.zeros( all_matrices.shape )
            for i in range(all_matrices.shape[1]):
                new_mat[:,i] = np.logical_or( np.logical_or( all_matrices[:,i], np.roll(all_matrices[:,i], 12) ), np.roll(all_matrices[:,i], -12) )
            all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.9/(1+ 0.2*(i%16) + 0.5*(i%8) + 0.5*(i%4) + 4*(i%2)):
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if register augmentation has been asked
    if len(register_aug) > 0:
        # do register augmentation
        print('performing register augmentation')
        new_all_matrices = np.array(all_matrices)
        for shift in register_aug:
            print('range shift: ', shift)
            new_mat = np.zeros( all_matrices.shape )
            for i in range(all_matrices.shape[1]):
                # print('i: ', i)
                if np.count_nonzero(all_matrices[:,i]) >= 1:
                    new_mat[:,i] = np.roll(all_matrices[:,i], shift)
            new_all_matrices = np.hstack( (new_all_matrices, new_mat) )
        all_matrices = np.array( new_all_matrices )
    
    # compute feature matrix
    print('computing tension array')
    f = compute_tension(all_matrices, normalize=True, smoothen=True)
    
    return all_matrices, f
# end get_concat_tension_np_from_folder

def get_concat_VL_np_from_folder(folderName, parts_for_surface, time_res, transpose = False, bin_out=True, voice_aug=True, register_aug=[], padding=32):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    if len(allDocs) is 0:
        print("Didn't find .xml file - looking for .mxl")
        allDocs = glob.glob(folderName + os.sep + "*.mxl")
    if len(allDocs) is 0:
        print("Didn't find .mxl file - looking for .mid")
        allDocs = glob.glob(folderName + os.sep + "*.mid")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        print('Running for ', fileName)
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    
    # pad-em-all
    if padding > 0:
        print('performing padding')
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # remove zero columns
            mask = [] # of columns TO KEEP
            for j in range( m.shape[1] ):
                if sum( m[:,j] ) != 0:
                    mask.append(j)
            m = m[ :, mask ]
            # make a padding matrix
            padder = np.zeros((128, padding))
            # pad
            all_matrices[i] = np.hstack( (padder, m) )
            if i == len(all_matrices)-1:
                all_matrices[i] = np.hstack( (all_matrices[i], padder) )
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    # check if register augmentation has been asked
    if len(register_aug) > 0:
        # do register augmentation
        print('performing register augmentation')
        new_all_matrices = np.array(all_matrices)
        for shift in register_aug:
            print('range shift: ', shift)
            new_mat = np.zeros( all_matrices.shape )
            for i in range(all_matrices.shape[1]):
                # print('i: ', i)
                if np.count_nonzero(all_matrices[:,i]) >= 1:
                    new_mat[:,i] = np.roll(all_matrices[:,i], shift)
            new_all_matrices = np.hstack( (new_all_matrices, new_mat) )
        all_matrices = np.array( new_all_matrices )

    # pseudo-voice augmentation
    if voice_aug:
        # construct "high voice" matrix
        m_h = np.zeros( all_matrices.shape )
        # construct "high-low voice" matrix
        m_hl = np.zeros( all_matrices.shape )
        for i in range( all_matrices.shape[1] ):
            # check if column has notes
            if np.sum(m_h[:,i]) > 0:
                # keep the index/pitch of the highest and lowest pitches
                highest_idx = np.max( np.nonzero( all_matrices[:,i] ) )
                lowest_idx = np.min( np.nonzero( all_matrices[:,i] ) )
                m_h[highest_idx, i] = 1
                m_hl[highest_idx, i] = 1
                m_hl[lowest_idx, i] = 1
        # contraints - or current step - as top layer
        top_layer = np.hstack( (m_h, m_hl) )
        # notes - or previous step - as bottom layer
        bottom_layer = np.hstack( (m_hl, all_matrices) )
        # remove first note from top and last from bottom
        top_layer = top_layer[:, 1:]
        bottom_layer = bottom_layer[:, :-1]
        all_matrices = np.vstack( (top_layer, bottom_layer) )
    return all_matrices
# end get_concat_VL_np_from_folder

def get_separate_tension_np_from_folder(folderName, parts_for_surface, time_res):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    if len(allDocs) is 0:
        print("Didn't find .xml file - looking for .mxl")
        allDocs = glob.glob(folderName + os.sep + "*.mxl")
    if len(allDocs) is 0:
        print("Didn't find .mxl file - looking for .mid")
        allDocs = glob.glob(folderName + os.sep + "*.mid")
    
    # keep names
    all_names = []
    # keep all matrices for each file
    all_matrices = []
    # keep all feature/tension arrays
    all_features = []
    
    for fileName in allDocs:
        print('Running for ', fileName)
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_names.append(fileName)
        all_matrices.append(m)
        f = compute_tension(m, smoothen=True, smoothingWindow=8)
        all_features.append(f)
    return all_names, all_matrices, all_features

# running for all files in folder but returns all pieces concatenated in a large matrix
def get_concat_rel_pcp_np_from_folder(folderName, parts_for_surface, time_res, parts_for_tonality, bin_out=True):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    # keep all tonalities
    all_tonalities = [];
    
    for fileName in allDocs:
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
        # get tonalities
        t, lt = get_parts_np_from_file(fileName, parts_for_tonality, time_res)
        all_tonalities.append(t)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    '''
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
    '''
    
    all_matrices = np.hstack(all_matrices)
    
    '''
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    '''
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    '''
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.3:
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # compute feature matrix
    print('computing feature matrix')
    f = compute_features(all_matrices)
    
    horizontal_range = 8
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, all_matrices.shape[1]) )
    for i in range(horizontal_range, all_matrices.shape[1]-horizontal_range):
        f[0,i] = np.mean( np.sum( all_matrices[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1 )
        tmpMat = all_matrices[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        f[1,i] = np.mean( np.sum(tmpMat_part, axis=0)/4.0 )
    '''
    
    return all_matrices, all_tonalities