# Deep Kernel Learning for Mortality Prediction in the Face of Temporal Shift
 
Code based on: https://github.com/YerevaNN/mimic3-benchmarks

We modified the code for extracting a population with temporal shift.

## Citation

Please cite the following publication:

Miguel Rios and Ameen Abu-Hanna. "Deep Kernel Learning for Mortality Prediction in the Face of Temporal Shift." AIME (2021). 

## Requirements

requirements_py3.6.txt

## Install

    python setup.py develop


## Build Dataset 

Examples to build the dataset in:

    prepro_dataset.sh

Example to extract temporal shift dataset.

1. first extract for each patient DB source and year
    
    python mimic3benchmark/scripts/split_carevue_and_metavision.py mimic3benchmark_testdata/all_stays.csv mimic3benchmark_testdata/split_stays.csv

2. split train and test given database source
    
    python mimic3benchmark/scripts/split_train_and_test_ood.py mimic3benchmark_testdata/ --split_stays mimic3benchmark_testdata/split_stays.csv --train carevue --test metavision    
    

## LSTM Baseline

Example to train the baseline:

    python -um mimic3models.in_hospital_mortality.train_lstm --dim 16 --timestep 1.0 --depth 1 --dropout 0.3 --batch_size 32 --data mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir lstm_shift --epoch 30 --lr 1e-3 --bidirectional

Example test baseline

    python -um mimic3models.in_hospital_mortality.testprob_lstm --dim 16 --timestep 1.0 --dropout 0.3 --batch_size 32 --data mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir lstm_shift/20-03-06.15h29m10s.wh2yofwf/ --aggregation_type mean --bidirectional --best_model lstm_shift/20-03-06.15h29m10s.wh2yofwf/best_model.pt  
    

# Deep Kernel Learning 

Examples to train the DKL model:

    python -um mimic3models.in_hospital_mortality.train_dkl_tmpshift --dim 16 --timestep 1.0 --dropout 0.3 --batch_size 64 --data mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir tmpshift_test --lr 0.01 --aggregation_type mean --epochs 20 --grid_size 64 --samples 8 --bidirectional --seed 1

Example to test the DKL model:

    python -um mimic3models.in_hospital_mortality.testprob_dkl_tmpshift --dim 16 --timestep 1.0 --dropout 0.3 --batch_size 64 --data mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir tmpshift_test --lr 0.01 --aggregation_type mean --epochs 20 --grid_size 64 --samples 8 --bidirectional --seed 1 --best_model tmpshift_test/21-03-22.14h54m22s.hev1crs5/best_model.pt  --best_lk tmpshift_test/21-03-22.14h54m22s.hev1crs5/best_lk.pt

