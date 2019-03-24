#xml as input, returns an xml with gct harmonic analysis and comments (for any double gcts for the same chord)
from music21 import *
# import itertools
from itertools import combinations
import numpy as np

class GCT:
    #id as a static variable maybe?
    def __init__(self, root, chType, chExtentions, chordForm):
        self.root = root
        self.type = chType
        self.extentions = chExtentions
        self.label = chordForm

def HARM_rootExtentionForm(shortest, chExtentions):
    #find first pitches
    #print("shortest: ", shortest)
    firstsPitches = []
    for i in range(len(shortest)):
        for j in range(len(shortest[i])):
            firstsPitches.append(shortest[i][j][0])
            pClList = []
            for k in range(len(shortest[0][0])):
                pCl = shortest[i][j][k] - shortest[i][j][0]
                if pCl <0:
                    pCl = pCl + 12
                pClList.append(pCl)
            shortest[i][j] = pClList

    #move extention relatively to 0
    for i in range(len(chExtentions)):
        for j in range(len(chExtentions[i])):
            chExtentions[i][j] = chExtentions[i][j] - firstsPitches[i]
            if chExtentions[i][j] <0:
                chExtentions[i][j] = chExtentions[i][j]+12

    #make final form of GCT chord notation
    for i in range(len(shortest)):
        shortest[i].insert(0,firstsPitches[i])
        shortest[i].append(sorted(chExtentions[i]))
    
    #if extention lower that the higest pitch of type, add 12
    notation = shortest
    for i in range(len(notation)):
        maxNot = max(notation[i][1])
        for j in range(len(notation[i][2])):
            if notation[i][2][j]<maxNot:
                notation[i][2][j] = notation[i][2][j] + 12
    
    #with np arrays
    '''for i in range(len(notation)):
        notation[i][0] = np.array(notation[1][0])
        notation[i][2] = np.array(notation[i][2])
        notation[i][1] = np.array(notation[i][1])
        j = notation[i][2]< np.max(notation[i][1])
        notation[i][2][j] = notation[i][2][j]+12'''
    #print("Chord: ", notation)
    return(notation, firstsPitches, pClList)

def HARM_shortestFormOfSubsets(maxConSubs):
    maxConSubss = list(maxConSubs)
    lastFirstInterval = [] #[[0]*len(maxConSubss)*len(maxConSubss[0])]
    shiftedChords = []

    for i in range(0, len(maxConSubss)):

        shiftedCh = []
        lastFirstInterval1 = []
        n = 0
        while n < len(maxConSubss[i]):
            shiftedCh1 = []

            maxConSubss[i].insert((len(maxConSubss[i])-1), maxConSubss[i].pop(0)) #circlular shifting
            shiftedCh1.extend(maxConSubss[i])#put it in shoftedCh1 array
            lastFirstInt = maxConSubss[i][-1]-maxConSubss[i][0] #find interval between first and last pitch of the max sconsonand chord
            if lastFirstInt<0: #if < 0 add 12 to move it to the next octave
                lastFirstInt = lastFirstInt + 12
            #lastFirstInterval[i][n] = lastFirstInt #put it in lastFirstInterval array
            lastFirstInterval1.append(lastFirstInt)
            shiftedCh.append(shiftedCh1) #put it in shiftesCh array
            n = n+1
        shiftedChords.append(shiftedCh)#put in the shifted chords array for each
        lastFirstInterval.append(lastFirstInterval1)

    #print("Intervals between first and last pitrch (for each max consonant): ", lastFirstInterval)
    
    shortestAll = []
    for i in range(len(lastFirstInterval)):
        shortestChOfEach = []
        shortest = min(lastFirstInterval[i])
        for j in range(len(lastFirstInterval[i])):
            if lastFirstInterval[i][j] == shortest:
                shortestChOfEach.append(shiftedChords[i][j]) #put it in an array
        shortestAll.append(shortestChOfEach) #make array with all shortest chords

    #NEEDS TESTING
    #if the shortest forms are more than one, you have to choose somehow
    for i in range(0, len(shortestAll)):
        for j in range(0, len(shortestAll[i])):
            #for k in range(0, len(shortestAll[i][j])):
            if len(shortestAll[i]) >=2: #if there are more than one shortest forms for one chord
                for j in range(len(shortestAll[i])-1):
                    baseLengthj = shortestAll[i][j][1]- shortestAll[i][j][0] #find the base length (first and second interval) of a shortest form of a chord
                    baseLengthjNext = shortestAll[i][j+1][1] - shortestAll[i][j+1][0] #find the base length of the next shortest form of a chord
                    if baseLengthj < 0:
                        baseLengthj = baseLengthj + 12 #if < 0 move it to the next octave
                    if baseLengthjNext < 0:
                        baseLengthjNext = baseLengthjNext + 12
                    if (baseLengthj) < (baseLengthjNext): #find the shortest form with shortest baselength
                        shortestBaseLength = shortestAll[i][j] #and that's the form I want
                        print("Shortest with minimum baselength:", shortestBaseLength) #NA TO TSEKARW ME POLLA CHORDS
                        #KAI PREPEI NA FTIAKSW ARRAY GIA OLA TA BASELENGTHS
    # max - messy workaround
    for i in range(len(shortestAll)):
        if len(shortestAll[i]) > 1:
            shortestAll[i] = [shortestAll[i][0]]
    return shortestAll

def HARM_findExtentions(m, maxConSubs):
    m = np.array(m)
    chEx = []
    for s in maxConSubs:
       chEx.append(sorted(list(m [[not (m[i] in s) for i in range(len(m))]])))
    return chEx

#keep only consonant subsets with max length
def HARM_findMaximalConsonantSubsets(consonant):
    maxConSubs = []
    for i in consonant:
        if len(i) == len(max(consonant,key=len)):
            maxConSubs.append(i)
    return maxConSubs

#find consonant intervals between pitches
def HARM_findConsonantSequencesOfSubsets(consWeights, subs):
    cons = [] #empty list
    for s in subs:
        #make a 2d list of zeros
        d = [[0]*len(s)]*(len(s))
        #make a second 2d list of zeros for appending ones
        #This is going to be the 2d array with the distances of notes of the chord
        #I use np to take the any func
        dBin = np.array([[0]*len(s)]*(len(s)))
        for i in range(0, len(s)):
            for j in range(0, len(s)):
                #find the distance between two pitches of the subset
                d[i][j]= abs(s[j]- s[i])
                #if the distance is negative, add 12
                while d[i][j] < 0: 
                    d[i][j] = d[i][j] + 12
                #if the consWeight is consonant
                if consWeights[d[i][j]]==1: 
                    #put an 1 into the dBin list
                    dBin[i][j] = 1
        # all values of the list equals one, the pitch sequence is consonant
        if np.all(dBin)==1:
            cons.append(s)
    return cons

def HARM_findSubsets(m):
    s = m
    #find all the possible compinations
    subsets = sum(map(lambda r: list(combinations(s, r)), range(1, len(s)+1)), [])
    
    #reversed to bring max length subset first
    subsRev = list(reversed(subsets))
    subs = []
    
    #sort the subsRev
    for i in subsRev:
        subs.append(sorted(i))
    #print("Subsets: ", subs)
    return subs

def HARM_findPitchClassesfromChord(chord):
    modChord = [i % 12 for i in chord] #modulo 12 to chord list to take the pitch classes
    return modChord

def HARM_takeOnlyUniqueValuesfromPitchClasses(modChord):
    m = list(set(modChord))
    return m

#function that returns the final form of chord
def HARM_consonanceChordRecognizer(chord, consWeights=[1,0,0,1,1,1,0,1,1,1,0,0]):

    #find the pitch classes from the original chord
    modChord = HARM_findPitchClassesfromChord(chord)

    #take only unique values from the pitch classes array
    m = HARM_takeOnlyUniqueValuesfromPitchClasses(modChord)

    #find subsets/possible combinations between pitches
    subs = HARM_findSubsets(m)
    
    #find consonant intervals between pitches
    consonant = HARM_findConsonantSequencesOfSubsets(consWeights, subs)

    #find Maximal Consonant Subsets
    maxConSubs = HARM_findMaximalConsonantSubsets(consonant)

    #find chord extentions
    chExtentions = HARM_findExtentions(m, maxConSubs)

    #find shortest form func
    shortest = HARM_shortestFormOfSubsets(maxConSubs)

    #chord label
    chordForm, root, chType = HARM_rootExtentionForm(shortest, chExtentions)
    condensed_gct = []
    for c in chordForm:
        tmpCondensed = np.append(np.array(c[0]), np.array(c[1]) )
        if c[2]:
            c[2] = sorted(c[2])
            tmpCondensed = np.append( tmpCondensed, np.array(c[2]) )
        condensed_gct.append( tmpCondensed )
    # print(chordForm)
    # return chordForm
    return condensed_gct, chordForm

def HARM_eliminate_foreigns(rtx, gcts, k, m):
    # make tonality pitch class arrat
    num_foreigns = []
    for g in rtx:
        # make chord pitch class array - from rtx form
        c = np.mod(g[0] + np.array(g[1]), 12)
        num_foreigns.append( len(c) - np.sum( np.isin(c, m) ) )
    num_foreigns = np.array( num_foreigns )
    min_foreigns = np.min( num_foreigns )
    idxs = num_foreigns == min_foreigns
    rtx_out = []
    gcts_out = []
    for i in range(len(idxs)):
        if idxs[i]:
            rtx_out.append( rtx[i] )
            gcts_out.append( gcts[i] )
    # if more than one, get the ones with the first foreign closer to the end
    if len( gcts_out ) > 0:
        rtx_out_end = []
        gcts_out_end = []
        foreign_idx = []
        for i in range( len(gcts_out) ):
            # make chord pitch class array - from rtx form
            c = np.mod(gcts_out[i][1:] + gcts_out[i][0], 12)
            # check if any foreign
            if any( ~np.isin(c, m) ):
                foreign_idx.append( np.where(~np.isin(c, m))[0][0] )
            else:
                foreign_idx.append( len(c) )
        foreign_idx = np.array( foreign_idx )
        min_foreign_idx = np.min( foreign_idx )
        idxs = foreign_idx == min_foreign_idx
        for i in range(len(idxs)):
            if idxs[i]:
                rtx_out_end.append( rtx_out[i] )
                gcts_out_end.append( gcts_out[i] )
        rtx_out = rtx_out_end
        gcts_out = gcts_out_end
    return rtx_out, gcts_out
# end HARM_eliminate_foreigns

def HARM_closed_position(rtx, gct):
    rtx_out = []
    gcts_out = []
    openness = []
    for g in gct:
        openness.append( g[-1] - g[1] )
    openness = np.array( openness )
    min_openness = np.min( openness )
    idxs = openness == min_openness
    for i in range(len(idxs)):
        if idxs[i]:
            rtx_out.append( rtx[i] )
            gcts_out.append( gct[i] )
    return rtx_out, gcts_out
# end HARM_closed_position

def HARM_shift_good_intervals(g):
    # enters only if g has three elements - two pitch classes
    if g[2]-g[1] in [1, 2, 5]:
        g[0] = np.mod( g[0]+g[2] , 12 )
        g[-1] = np.mod( 12-g[-1] , 12 )
    return g
# end HARM_shift_good_intervals

def get_singe_GCT_of_chord(c, k=0, m=np.array([0,2,4,5,7,9,11])):
    all_gcts, rtx_form = HARM_consonanceChordRecognizer(c)
    # fix roots in key
    for i in range( len(all_gcts) ):
        all_gcts[i][0] = (all_gcts[i][0] - k)%12
        rtx_form[i][0] = (rtx_form[i][0] - k)%12
    # if more than one GCTs
    if len(all_gcts) > 1:
        # 1 & 2) get the ones with the smallest number of foreign pitches in the type
        # if more than one, get the ones with the first foreign closer to the end
        rtx_form, all_gcts = HARM_eliminate_foreigns(rtx_form, all_gcts, k, m)
    # 3) closed position
    if len(all_gcts) > 1:
        rtx_form, all_gcts = HARM_closed_position(rtx_form, all_gcts)
    # if still more than one, write it down
    if len(all_gcts) > 1:
        with open("GCT_logging.txt", "a") as myfile:
            myfile.write("chord: "+str(c)+'\n')
            myfile.write("gcts found: "+'\n')
            for g in all_gcts:
                myfile.write(str(g)+'\n')
    final_gct = all_gcts[0]
    # if only two pitch classes, shift to "good" intervals
    if len( final_gct ) == 3:
        final_gct = HARM_shift_good_intervals( final_gct )
    # if len(all_gcts) > 1:
    #     # 2) keep the one with the smallest root value
    #     all_roots = []
    #     for i in rtx_form:
    #         all_roots.append( rtx_form[0] )
    #     min_root = min(all_roots)
    #     for i in range(len(rtx_form)):
    #         if rtx_form[i][0] == min_root:
    #             final_gct = all_gcts[i]
    #             break
    # if np.array_equal( final_gct, np.array([0,0,3,7,10]) ):
    #     print('GATHCA')
    #     print('c: ', c)
    #     print('k: ', k)
    #     print('m: ', m)
    return final_gct
# end get_singe_GCT_of_chord

def get_singe_GCT_of_m21chord(c, k=0, m=np.array([0,2,4,5,7,9,11])):
    # get pitches of chord structure
    a = []
    for p in c.pitches:
        a.append( p.midi )
    # run function above
    final_gct = get_singe_GCT_of_chord(a, k, m)
    return final_gct