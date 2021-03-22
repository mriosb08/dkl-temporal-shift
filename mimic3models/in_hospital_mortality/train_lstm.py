import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
from mimic3models.models.lstm_model import LSTMClassifier
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.imbalanced import ImbalancedDatasetSampler
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


#torch.manual_seed(42)
#np.random.seed(42)

def eval_model(model, dataset, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        y_true = []
        predictions = []
        for data, labels in dataset:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            probs = sigmoid(logits)
            #_, predicted = torch.max(probs.data, 1)
            #y_hat_class = np.where(probs.data<0.5, 0, 1)
            predictions += [p.item() for p in probs]#y_hat_class.squeeze()
            y_true += [y.item() for y in labels]
    #print(predictions)
    #print(y_true)
    clf_score = brier_score_loss(y_true, predictions, pos_label=1)
    logging.info("Brier score: %1.3f" % (clf_score))
    results = metrics.print_metrics_binary(y_true, predictions, logging)
    return results

def main(args):
    # define trainning and validation datasets
    args.mode = 'train'
    hidden_size = args.dim
    dropout = args.dropout
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    emb_size = args.emb_size
    aggregation_type = args.aggregation_type
    imbalance = args.imbalance
    bidirectional_encoder = args.bidirectional # TODO add into args
    seed = args.seed
    depth = args.depth
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")   
    # 1. Get a unique working directory 
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)  # make sure base_dir exists
    output_dir = tempfile.mkdtemp(prefix=datetime.now().strftime("%y-%m-%d.%Hh%Mm%Ss."), dir=base_dir)
    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'run.log')),
                logging.StreamHandler()
            ])
    
    logging.info('Workspace: %s', output_dir)
 

    
    if args.small_part:
        args.save_every = 2**30

    target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

    discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
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
    
    if imbalance:
        train_dataset = utils.MIMICDataset(train_reader, discretizer, normalizer, batch_labels=False)
        sampler = ImbalancedDatasetSampler(train_dataset)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    else:
        train_dataset = utils.MIMICDataset(train_reader, discretizer, normalizer, batch_labels=True)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = utils.MIMICDataset(val_reader, discretizer, normalizer, batch_labels=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #[B, M, feat_size]
    feat_size = train_dataset.data.shape[-1] 
    if target_repl:
        raise NotImplementedError("target repl not implemented")
        #T = train_raw[0][0].shape[0]
        #train_raw = extend_labels(train_raw)
        #val_raw = extend_labels(val_raw)


    # Define the classification model.
    model = LSTMClassifier(tag_size=1, #binary
                    feat_size= feat_size, 
                    hidden_size=hidden_size,
                    emb_size=emb_size,
                    bidirectional=bidirectional_encoder,
                    dropout=dropout,
                    depth=depth,
                    aggregation_type=aggregation_type)

    model = model.to(device)
    logging.info(args)
    logging.info(model)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if target_repl:
        raise  NotImplementedError("loss for each time step to be implemented") 
    else:
        criterion = nn.BCEWithLogitsLoss()

    # path to best model save on disk
    best_model = os.path.join(output_dir, 'best_model.pt') 
    best_val_auc = 0.

    results = []

    step = 0
    num_batches = 0

    # loop over the epochs
    for epoch_num in range(1, num_epochs+1): 
        loss_batch = .0
        num_batches = 0
        # loop over mini-batches
        for x, labels in train_dl:
            x = x.to(device)
            labels = labels.to(device)
            # Model is in training mode (for dropout).
            model.train()
            optimizer.zero_grad()
       
            # run forward
            logits = model(x)
            if imbalance:
                logits = logits.squeeze()
            loss = criterion(logits, labels)
            
            loss_batch += loss.item()
            # Backpropagate and update the model weights.
            loss.backward()
            optimizer.step()
            #loss_batch += loss.item()        
            num_batches += 1
        
            # Every 100 steps we evaluate the model and report progress.
            if step % args.steps == 0:
                logging.info("epoch (%d) step %d: training loss = %.2f"% 
                 (epoch_num, step, loss_batch/num_batches))
            
            
            step += 1
        
        
        metrics_results = eval_model(model,
                                    val_dl,
                                    device)
        metrics_results['epoch'] = epoch_num
        results.append(metrics_results)
        if metrics_results['auroc'] > best_val_auc:
            best_val_auc = metrics_results['auroc']
            # save best model in disk
            torch.save(model.state_dict(), best_model)
            logging.info('best model AUC of ROC = %.3f'%(best_val_auc))
            logging.info("Finished epoch %d" % (epoch_num))
            


    pickle.dump(results, open(os.path.join(output_dir, 'metrics.pkl'), "wb" ) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train')
    parser.description = 'train in hospital mortality classifier'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    common_utils.add_common_arguments(parser)
    parser.add_argument('--target_repl_coef', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=100, help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--emb_size', type=int, default=128, help='emb_size default:128')
    parser.add_argument('--imbalance', dest='imbalance', action='store_true')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
    args = parser.parse_args()
    main(args)

#example
#python -um mimic3models.in_hospital_mortality.train_lstm --dim 16 --emb_size 16 --timestep 1.0 --dropout 0.3 --batch_size 16 --data ../mimic3benchmark_tem    poraldata/in-hospital-mortality/  --output_dir ../lstm_temp_exp2 --lr 0.001 --aggregation_type mean --bidirectional  --epochs 30
