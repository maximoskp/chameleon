#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 19:05:10 2018

@author: maximoskaliakatsos-papakostas
"""

import music21 as m21

def generate_midi(sc, fileName="test_midi.mid", destination="/Users/maximoskaliakatsos-papakostas/Documents/python/miscResults/"):
    # we might want the take the name from uData information, e.g. the uData.input_id, which might preserve a unique key for identifying which file should be sent are response to which user
    mf = m21.midi.translate.streamToMidiFile(sc)
    mf.open(destination + fileName, 'wb')
    mf.write()
    mf.close()
# end generate_midi
def generate_xml(sc, fileName="test_xml.xml", destination="/Users/maximoskaliakatsos-papakostas/Documents/python/miscResults/"):
    mf = m21.musicxml.m21ToXml.GeneralObjectExporter(sc)
    mfText = mf.parse().decode('utf-8')
    f = open(fileName, 'w')
    f.write(mfText.strip())
    f.close()
# end generate_xml
def generate_abc(sc, fileName="test_abc.mid", destination="/Users/maximoskaliakatsos-papakostas/Documents/python/miscResults/"):
    # write temporary xml file
    mf = m21.musicxml.m21ToXml.GeneralObjectExporter(sc)
    mfText = mf.parse().decode('utf-8')
    f = open(fileName+"xml", 'w')
    f.write(mfText.strip())
    f.close()
    xml2abc.main(fileName+"xml")
# end generate_abc