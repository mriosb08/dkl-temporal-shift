#!/bin/bash
python -um mimic3benchmark.scripts.extract_subjects /data/workspace/physionet.org/files/mimiciii/1.4/ mimic3benchmark_data
python -um mimic3benchmark.scripts.validate_events mimic3benchmark_data
python -um mimic3benchmark.scripts.extract_episodes_from_subjects mimic3benchmark_data
python -um mimic3benchmark.scripts.split_train_and_test mimic3benchmark_data
python -um mimic3benchmark.scripts.create_in_hospital_mortality mimic3benchmark_data mimic3benchmark_data/in-hospital-mortality/
python -um mimic3models.split_train_val mimic3benchmark_data/in-hospital-mortality/
