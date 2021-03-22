from __future__ import absolute_import
import numpy as np
import sys
from mimic3benchmark.readers_conll import InHospitalMortalityReader
import re
import sys

reader = InHospitalMortalityReader(dataset_dir='../mimic3benchmark_textdata/in-hospital-mortality/train',
                              notes_dir='../mimic3benchmark_textdata/train',  
                              listfile='../mimic3benchmark_textdata/in-hospital-mortality/val_listfile.csv')

#print(reader.read_example(1))
N = reader.get_number_of_examples()
pos_total = 0
neg_total = 0
regex_flag = False
pos = 0
neg = 0
phrase = ' cmo '#' comfort measures only '#' cmo '

for n in range(N):
    patient = reader.read_example(n)
    patient_notes = patient['text']
    patient_info = patient['text_info']
    y = patient['y']

    if y == 1:
        pos_total += 1
    elif y == 0:
        neg_total += 1

    for doc, sentences in patient_notes.items():

        #print('doc', doc)
        #print(info[doc])
        #print('sentences', sentences)
        for sentence in sentences:
            sent = ' '.join(sentence)
            if re.search(phrase, sent):
                regex_flag = True
                print(sent, patient_info[doc], y)
    if regex_flag and y==1:
        pos +=1
        regex_flag = False
    elif regex_flag and y==0:
        neg += 1
        regex_flag = False

print('total pos', pos_total)
print('total neg', neg_total)
print('regex in pos (mortality)', pos)
print('regex in neg', neg)



