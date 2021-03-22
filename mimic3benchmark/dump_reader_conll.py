from __future__ import absolute_import
import numpy as np
import sys
from mimic3benchmark.readers_conll import InHospitalMortalityReader

reader = InHospitalMortalityReader(dataset_dir='mimic3_textdata/in-hospital-mortality/train',
                              notes_dir='mimic3_textdata/train',  
                              listfile='mimic3_textdata/in-hospital-mortality/train/listfile.csv')

#print(reader.read_example(1))
#patient = reader.read_example(int(sys.argv[1]))
#patient_notes = patient['text']
#x = patient['X']
#info = patient['text_info']
#print(len(x))
#print(len(x[0]))
#print(patient['name'])
#print(patient_notes)
N = reader.get_number_of_examples()
for i in range(N):
    patient = reader.read_example(i)
    patient_notes = patient['text']
    for doc, sentences in patient_notes.items():
        #print('doc', doc)
        #print(info[doc])
        for s in sentences:
            print(' '.join(s))


