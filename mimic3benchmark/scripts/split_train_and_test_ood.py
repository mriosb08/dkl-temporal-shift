from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import argparse
import csv


def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)


def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets for OOD. With DBSource')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('split_stays', type=str, help='list to split given a BD source or year')
    parser.add_argument('train', type=str, help='train dataset give BD name e.g. carevue')
    parser.add_argument('test', type=str, help='train dataset give BD name e.g. metavision')
    args, _ = parser.parse_known_args()

    train_set = set()
    test_set = set()

    with open(args.split_stays, "r") as test_set_file:
        csv_reader = csv.reader(test_set_file, delimiter=',')
        # the below statement will skip the first row
        next(csv_reader)
        for subject_id, db_source, year in csv_reader:
            if db_source == args.train:
                train_set.add(subject_id)
            elif db_source == args.test:
                test_set.add(subject_id)

    folders = os.listdir(args.subjects_root_path)
    folders = list((filter(str.isdigit, folders)))
    # some patients are in both DB 
    train_set_noisec = train_set - test_set
    test_set_noisec = test_set - train_set
    train_patients = [x for x in folders if x in train_set_noisec]
    test_patients = [x for x in folders if x in test_set_noisec]
    print('train', len(train_patients))
    print('test', len(test_patients))
    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")


if __name__ == '__main__':
    main()
