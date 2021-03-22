from __future__ import absolute_import
import numpy as np
import sys
from mimic3benchmark.readers_conll import InHospitalMortalityReader

reader = InHospitalMortalityReader(dataset_dir='../mimic3benchmark_textdata/in-hospital-mortality/train',
                              notes_dir='../mimic3benchmark_textdata/train',  
                              listfile='../mimic3benchmark_textdata/in-hospital-mortality/train/listfile.csv')

#print(reader.read_example(1))
N = reader.get_number_of_examples()
for n in range(N):
    patient = reader.read_example(n)
    patient_notes = patient['text']
    for doc, sentences in patient_notes.items():

        #print('doc', doc)
        #print(info[doc])
        #print('sentences', sentences)
        for sentence in sentences:
            print(' '.join(sentence))


