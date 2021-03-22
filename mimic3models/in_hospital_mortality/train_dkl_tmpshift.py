import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
import sys
from mimic3models.models.dkl_model import DKLModel
from mimic3models.models.lstm_tmpshift_model import LSTMFeatExtractor
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import common_utils
import numpy as np
from mimic3models.imbalanced import ImbalancedDatasetSampler
import gpytorch
import logging
import tempfile
import shutil
import pickle
from datetime import datetime
from sklearn.metrics import brier_score_loss

#torch.manual_seed(42)
#np.random.seed(42)



def eval_model(model, likelihood, dataset, device, eval_samples=1): #eval_samples=1
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(eval_samples):
        y_true = []
        predictions = []
        for data, labels in dataset:
            data = data.to(device)
            labels = labels.to(device)
            output_model = model(data)
            output = likelihood(output_model)
            #test_preds = model(data) #ge(0.5).float()
            #print(test_preds.mean)
            #print(test_preds.mean.size())
            #output = likelihood(test_preds)
            #print(output.probs.size())
            #sys.exit(0)
            # mean of samples
            probs = output.probs.mean(0)
            probs = probs[:, 1]
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
    grid_size = args.grid_size
    grid_dim = args.grid_dim
    samples = args.samples
    imbalance = args.imbalance
    aggregation_type = args.aggregation_type
    bidirectional_encoder = args.bidirectional # TODO add into args
    depth = args.depth
    seed = args.seed
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
    #shutil.copy(os.path.abspath(__file__), output_dir)
    
    if args.small_part:
        args.save_every = 2**30

    #target_repl = (args.target_repl_coef > 0.0 and args.mode == 'train')


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
    #args_dict['target_repl'] = target_repl

    # Read data
    train_dataset = utils.MIMICDataset(train_reader, discretizer, normalizer)
    if imbalance:
        sampler = ImbalancedDatasetSampler(train_dataset)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = utils.MIMICDataset(val_reader, discretizer, normalizer)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #[B, M, feat_size]
    feat_size = train_dataset.data.shape[-1] 


    
    # Define the feature extractor
    feature_extractor = LSTMFeatExtractor(feat_size= feat_size,
                                    emb_size=emb_size,
                                    hidden_size=hidden_size, 
                                    bidirectional=bidirectional_encoder,
                                    dropout=dropout,
                                    depth=depth,
                                    aggregation_type=aggregation_type)
    num_features = hidden_size * 2 if bidirectional_encoder else hidden_size 

    model = DKLModel(feature_extractor, num_dim=num_features, grid_size=grid_size)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, num_classes=2) 
    #likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    model = model.to(device)
    likelihood = likelihood.to(device)
    logging.info(args)
    logging.info(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adam([
    #{'params': model.gp_layer.hyperparameters()},
    #{'params': model.gp_layer.variational_parameters()},
    #{'params': model.feature_extractor.parameters()},
    #{'params': likelihood.parameters()},
    #], lr=learning_rate)
    #scheduler = MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_dataset))

    # path to best model save on disk
    best_model = os.path.join(output_dir, 'best_model.pt') 
    best_lk = os.path.join(output_dir, 'best_lk.pt')
    best_val_auc = 0.
    results = []
    step = 0
    num_batches = 0

    # loop over the epochs
    for epoch_num in range(1, num_epochs+1): 
        with gpytorch.settings.use_toeplitz(False):
            loss_batch = .0
            num_batches = 0
            model.train()
            likelihood.train()
            # loop over mini-batches
            with gpytorch.settings.num_likelihood_samples(samples): #10
                for x, labels in train_dl:
                    x = x.to(device)
                    labels = labels.to(device)
                    # Model is in training mode (for dropout).
                
                    optimizer.zero_grad()
       
                    # run forward
                    output = model(x)
                    loss = -mll(output, labels).mean()
                    # Backpropagate and update the model weights.
                    loss.backward()
                    optimizer.step()

                    loss_batch += loss.item() 
                    num_batches += 1
                    if step % args.steps == 0:
                        logging.info("epoch (%d) step %d: training mll = %.2f"% 
                                (epoch_num, step, loss_batch/num_batches))
            
                    step += 1
            #scheduler.step()
            metrics_results = eval_model(model,
                                       likelihood,
                                        val_dl,
                                        device,
                                        eval_samples=samples)
            metrics_results['epoch'] = epoch_num
            results.append(metrics_results)
            if metrics_results['auroc'] > best_val_auc:
                best_val_auc = metrics_results['auroc']
                # save best model in disk
                torch.save(model.state_dict(), best_model)
                torch.save(likelihood.state_dict(), best_lk)
                logging.info('best model AUC of ROC = %.3f'%(best_val_auc))
            logging.info("Finished epoch %d" % (epoch_num))
            


    pickle.dump(results, open(os.path.join(output_dir, 'metrics.pkl'), "wb" ) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train')
    parser.description = 'train in hospital mortality DKL classifier'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    common_utils.add_common_arguments(parser)
    parser.add_argument('--steps', type=int, default=300, help='perfom evaluation and model selection on validation default:100')
    parser.add_argument('--grid_size', type=int, default=100, help='grid size = inducing points of GP default:100')
    parser.add_argument('--samples', type=int, default=1, help='samples = likelihood samples of GP default:1')
    parser.add_argument('--grid_dim', type=int, default=1, help='grid dim = inducing points of GP default:5')
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/in-hospital-mortality/'))
    parser.add_argument('--imbalance', dest='imbalance', action='store_true')
    parser.add_argument('--emb_size', type=int, default=64,
                          help='embedding size default 64') 
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='dkl_exp')
    args = parser.parse_args()
    main(args)
#eg command
#python -um mimic3models.in_hospital_mortality.train_dkl_tmpshift --dim 16 --timestep 1.0 --dropout 0.3 --batch_size 64 --data ../mimic3benchmark_temporaldata/in-hospital-mortality/  --output_dir ../tmpshift_test --lr 0.01 --aggregation_type mean --epochs 20 --grid_size 64 --samples 8 --bidirectional --seed 1