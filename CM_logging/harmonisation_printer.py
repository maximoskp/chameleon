#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:30:17 2019

@author: maximoskaliakatsos-papakostas
"""

import matplotlib.pyplot as plt

def initialise_log_file(file_name):
    with open(file_name+'.txt', 'w') as the_file:
        the_file.write('================== Logging initialisation ================== \n')
        the_file.write('Initialising file: ' + file_name+'.txt' + '\n')
# end initialise_log_file
def print_log_line( file_name, line_str ):
    with open(file_name+'.txt', 'a') as the_file:
        the_file.write( line_str + '\n')
# end print_log_line
def print_image_with_axis(file_name, mat_to_print, x_labels, y_labels, vertical_x=False):
    tmpfig = plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(mat_to_print, cmap='gray_r', interpolation='none');
    if vertical_x:
        plt.xticks(range(len(x_labels)), x_labels, rotation='vertical')
    else:
        plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    tmpfig.savefig(file_name + '.png', format='png', dpi=300, bbox_inches="tight")
    plt.clf()
# end print_image_with_axis
def print_image_and_numbers_with_axis(file_name, mat_to_print, nums_to_print, x_labels, y_labels, vertical_x=False):
    tmpfig = plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(mat_to_print, cmap='gray_r', interpolation='none')
    # plot numbers
    for i in range( nums_to_print.shape[0] ):
        for j in range( nums_to_print.shape[1] ):
            plt.text( j-0.5, i+0.2, str( nums_to_print[i,j] ), color=[1,0,0] )
    if vertical_x:
        plt.xticks(range(len(x_labels)), x_labels, rotation='vertical')
    else:
        plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), y_labels)
    tmpfig.savefig(file_name + '.png', format='png', dpi=300, bbox_inches="tight")
    plt.clf()
# end print_image_with_axis