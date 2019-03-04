import numpy as np

def computeDICfromMIDI(cIN1, cIN2):
    cIN1 = np.array(cIN1).astype(int)
    cIN2 = np.array(cIN2).astype(int)

    c1 = np.zeros(12)
    c2 = np.zeros(12)

    c1[cIN1%12] = 1
    c2[cIN2%12] = 1

    d = np.zeros(12)
    dIDs = np.array([0,1,-1,2,-2,3,-3,4,-4,5,-5,6])
    #comment for hot fix
    #fix22
    for i in range(0, len(c1)):
        if c1[i] > 0:
            for j in range(0, len(c2)):
                if c2[j] > 0:
                    m = j-i
                    if m > 6:
                        m = m - 12
                    elif m <= -6:
                        m = 12 + m
                    d[dIDs == m] = d[dIDs == m] + 1
    return d, dIDs

def computeDICsfromChordList(chordsList):
    dics = np.zeros((len(chordsList) - 1, 12))
    for i in range(len(chordsList)-1):
        tmpDIC = computeDICfromMIDI(chordsList[i], chordsList[i+1])
        dics[i,:] = tmpDIC
    return dics
