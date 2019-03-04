#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 08:28:19 2018

@author: maximoskaliakatsos-papakostas
"""

import matplotlib.pyplot as plt
import NN_polyphonic as nnp

model = nnp.PolyFiller()
model.fill_notes_in_matrix(num_notes=32)

plt.imshow(model.matrix)