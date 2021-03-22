import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
import gpytorch
from mimic3models.models.dkl_model import DKLModel
from mimic3models.models.lstm_tmpshift_model import LSTMFeatExtractor
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils
import numpy as np
import logging
import tempfile
import shutil
import pickle
from datetime import datetime
from sklearn.metrics import brier_score_loss


torch.manual_seed(42)
np.random.seed(42)

def eval_model(model, likelihood, dataset, device, eval_samples=50):
    model.eval()
    likelihood.eval()
    #sigmoid = nn.Sigmoid()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(eval_samples):
        #print(gpytorch.settings)
        y_true = []
        predictions = []
        logs = []
        for data, labels in dataset:
            data = data.to(device)
            labels = labels.to(device)
            output = likelihood(model(data))
            # mean of batch
            #print(dir(output))
            #print(model(data))
            #print(dir(model(data)))
            #print(output.logits.mean(0))
            #print(output.probs.size())
            
            probs = output.probs.mean(0)
            probs = probs[:, 1]
            logits = output.logits.mean(0)
            logits = logits[:, 1]
            #pred = output.probs.mean(0).argmax(-1)
            #print(pred)
            predictions += [p.item() for p in probs]#y_hat_class.squeeze()
            logs += [l.item() for l in logits]
            y_true += [y.item() for y in labels]
    #print(predictions)
    #print(y_true)
    #clf_score = brier_score_loss(y_true, predictions, pos_label=1)
    #logging.info("Brier score: %1.3f" % (clf_score))
    results = metrics.print_metrics_binary(y_true, predictions, logging)
    return results, predictions, y_true, logs

def main(args):
    # define trainning and validation datasets
    args.mode = 'test'
    hidden_size = args.dim
    dropout = args.dropout
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    emb_size = args.emb_size
    best_model = args.best_model
    best_lk = args.best_lk
    grid_size = args.grid_size
    aggregation_type = args.aggregation_type
    samples = args.samples
    bidirectional_encoder = args.bidirectional # TODO add into args
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")   
    # 1. Get a unique working directory 
    output_dir = args.output_dir
    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')
    

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                         listfile=os.path.join(args.data, 'test_listfile.csv'),
                                         period_length=48.0)

    
    discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

    discretizer_header = discretizer.transform(test_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)

    args_dict = dict(args._get_kwargs())
    args_dict['header'] = discretizer_header
    args_dict['task'] = 'ihm'
    args_dict['target_repl'] = target_repl

    # Read data
    test_dataset = utils.MIMICDataset(test_reader, discretizer, normalizer, batch_labels=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #[B, M, feat_size]
    feat_size = test_dataset.data.shape[-1] 


    # Define the classification model.
    feature_extractor = LSTMFeatExtractor(feat_size= feat_size,
                                    emb_size=emb_size,
                                    hidden_size=hidden_size, 
                                    bidirectional=bidirectional_encoder,
                                    dropout=dropout,
                                    aggregation_type=aggregation_type)
    num_features = hidden_size * 2 if bidirectional_encoder else hidden_size 

    model = DKLModel(feature_extractor, num_dim=num_features, grid_size=grid_size)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, num_classes=2) #gpytorch.likelihoods.BernoulliLikelihood() 


    model.load_state_dict(torch.load(best_model))
    likelihood.load_state_dict(torch.load(best_lk))
    logging.info(model)
    model = model.to(device)
    likelihood = likelihood.to(device)

    metrics_results, pred_probs, y_true, logs = eval_model(model,
                                likelihood,
                                test_dl,
                                device)
            

    
    pickle.dump(metrics_results, open(os.path.join(output_dir, 'test_metrics.pkl'), "wb" ) )
    pickle.dump(pred_probs, open(os.path.join(output_dir, 'test_predprobs.pkl'), "wb" ) )
    pickle.dump(logs, open(os.path.join(output_dir, 'test_predlogits.pkl'), "wb" ) )
    pickle.dump(y_true, open(os.path.join(output_dir, 'test_ytrue.pkl'), "wb" ) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test')
    parser.description = 'test DKL in hospital mortality classifier'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=100, help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--grid_size', type=int, default=100, help='grid size = inducing points of GP default:100')
    parser.add_argument('--emb_size', type=int, default=128, help='emb_size default:128')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
    parser.add_argument('--samples', type=int, default=1, help='samples = likelihood samples of GP default:1')
    parser.add_argument('--best_model', type=str, required=True, help='best model path')
    parser.add_argument('--best_lk', type=str, required=True, help='best model path')
    args = parser.parse_args()
    main(args)

#example
#python -um mimic3models.in_hospital_mortality.testprob_dkl_tmpshift --dim 16 --timestep 1.0 --dropout 0.3 --batch_size 64 --data ../mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir ../tmpshift_test --lr 0.01 --aggregation_type mean --epochs 20 --grid_size 64 --samples 8 --bidirectional --seed 1 --best_model ../tmpshift_test/21-03-22.14h54m22s.hev1crs5/best_model.pt  --best_lk ../tmpshift_test/21-03-22.14h54m22s.hev1crs5/best_lk.pt
