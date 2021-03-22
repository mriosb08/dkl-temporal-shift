import sys
import os
import pandas as pd 

def main(args):
    subjects_root_path, output_file = args
    #folders = os.listdir(subjects_root_path)
    df = pd.read_csv(subjects_root_path)
    #SUBJECT_ID,HADM_ID,ICUSTAY_ID,LAST_CAREUNIT
    df_stay = df[['SUBJECT_ID','DBSOURCE', 'INTIME']]
    df_stay['INTIME'] = pd.to_datetime(df_stay['INTIME'])
    df_stay['INTIME'] = df_stay['INTIME'].dt.year
    print(df_stay)
    df_stay.to_csv(output_file, index=False) 
    return

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('split.py [path_data] [output_file]')
    else:
        main(sys.argv[1:])

