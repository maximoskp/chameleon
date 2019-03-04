#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 00:13:10 2018

@author: maximoskaliakatsos-papakostas
"""

import os
cwd = os.getcwd()
import sys
sys.path.insert(0, cwd+'/CM_blending')
import CM_BlendingSession as bld

mainFolder = cwd + '/trained_idioms/'
idioms_list = []

for file in os.listdir(mainFolder):
    if file.endswith(".pickle"):
        idioms_list.append( file.split('.')[0] )
print(idioms_list)
bld.blend_all_idioms_from_list( idioms_list )

# for f in os.walk(mainFolder):
#     if f[2][0].endswith('.xml'):
#         print('f: ', f[0].split('/')[-1])
#         idiom = tic.TrainingIdiom(f[0]+'/')
#         idiom = grp.group_gcts_of_idiom(idiom)
#         # save learned idiom
#         with open('trained_idioms/'+idiom.name+'.pickle', 'wb') as handle:
#             pickle.dump(idiom, handle, protocol=pickle.HIGHEST_PROTOCOL)