#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:04:32 2018

@author: maximoskaliakatsos-papakostas

"""

import numpy as np

# auxiliary functions

def str2np(s):
    return np.fromstring(s[1:-1], sep=' ')
# end str2np

def mode2relpcp(m):
    return np.histogram( m, bins=range(13) )[0]

def np2str(np_in):
    # removes undesired spaces
    tmp_string = np.array2string(np_in)
    tmp_string = tmp_string.replace('  ', ' ')
    tmp_string = tmp_string.replace('[ ', '[')
    return tmp_string
# end np2str

# function for getting all the offsets 
def get_offsets(array_in):
    offsets = []
    for i in array_in:
        offsets.append(i.offset)
    return offsets
# end get_offsets

def gct2relpcp(g):
    r_pc = np.mod(g[0] + g[1:], 12)
    return np.histogram( r_pc, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12] )[0]
# end gct2relpcp

def array2relpcp(a):
    r_pc = np.mod(a, 12)
    return np.histogram( r_pc, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12] )[0]
# end array2relpcp