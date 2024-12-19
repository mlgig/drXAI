import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, dataset_class, Data_Verifier
from Models.model import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss import get_loss_module
from Models.utils import load_model
from Training import SupervisedTrainer, train_runner
from copy import deepcopy
from aeon.datasets import load_from_tsfile
#from ..utils import load_data_ConvTran
from torch.cuda import  empty_cache as empty_gpu_cache
import timeit

data_path = "/media/davide/DATA/datasets/Multivariate2018_ts/others_new/" #"/mnt/storage/davides/channel_selection/Multivariate_ts/"















from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from utils import dataset_class
from torch.utils.data import DataLoader
import numpy as np

def load_data_ConvTran(X_train, X_test, y_train, y_test, val_ratio=0.25, batch_size=16, to_half=False):

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # assuming equal length data
    _ , n_channels , seq_len = X_train.shape

    train_data, train_label, val_data, val_label = split_dataset(X_train, y_train,val_ratio)

    # creating loaders
    train_dataset = dataset_class(train_data, train_label,to_half)
    val_dataset = dataset_class(val_data, val_label, to_half)
    dev_dataset = dataset_class( X_train, y_train,to_half)
    test_dataset = dataset_class(X_test,y_test, to_half)
    if to_half:
        batch_size=1

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_loader, val_loader, dev_dataset, test_loader

def split_dataset(data, label, validation_ratio):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio) #, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label













logger = logging.getLogger('__main__')
"""
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.25, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")
# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=12, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=6, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')
args = parser.parse_args()
"""

# TODO modify this main s.t. it will be callable train and inference methods
# TODO figure out what to do with EigenWorms dataset
# TODO move following function somewhere else?
# TODO don't save the model during validation
# TODO add + 1 / -1 if odd for emb size

def build_init_model(config,n, shape, n_labels, float16=False):
    logger.info("Creating model ...")
    config['Data_shape'] = shape
    config['num_labels'] = n_labels
    if float16:
        config['lr']/=10
        config['batch_size']=1

    model = model_factory(config)
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], problem +"_"+str(n) + '.pth'.format('last'))
    model = model.half() if float16 else model
    model.to(device)
    return model, save_path





config = {
    'data_path': 'Dataset/UEA/', 'Norm': False,  'val_ratio': 0.25, 'print_interval': 10, 'Net_Type': ['C-T'],
    'emb_size': 12, 'dim_ff': 256, 'num_heads': 6,   'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE',
    'epochs': 1,'batch_size': 16, 'lr': 0.001, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'accuracy',
    'gpu': 0,  'console': False, 'output_dir': 'Results/Dataset/UEA/',
}



def build_train_model(train_loader,val_loader, dev_dataset):
    tensorboard_writer = SummaryWriter('summary')
    shape, n_labels = X_train.shape, np.unique(y_train).shape[0]
    to_half = True if train_loader.dataset.to_half else False

    model, save_path = build_init_model(config, n_tral, shape, n_labels, to_half)
    # ---------------------------------------------- Validating The Model ------------------------------------
    logger.info('Starting training...')
    trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
                                print_interval=config['print_interval'], console=config['console'],
                                print_conf_mat=False)
    val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
                                      print_interval=config['print_interval'], console=config['console'],
                                      print_conf_mat=False)
    best_n_epochs, _ = train_runner(config, model, trainer, save_path, val_evaluator=val_evaluator)

    # clean what used so far
    del train_loader ; del val_loader ; del model
    empty_gpu_cache()
    # ---------------------------------------------- Final Training ------------------------------------
    final_config = deepcopy(config)
    final_config['epochs'] = best_n_epochs
    final_config['emb_size'] = np.ceil(config['emb_size'] / 0.75).astype(int)  # TODO hard coded
    final_config['num_heads'] = np.ceil(config['num_heads'] / 0.75).astype(int)
    model, _ = build_init_model(final_config, n_tral, shape, n_labels)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=final_config['batch_size'], shuffle=True, pin_memory=True)
    final_trainer = SupervisedTrainer(model, dev_loader, device, final_config['loss_module'], final_config['optimizer'],
                                      l2_reg=0, print_interval=final_config['print_interval'],
                                      console=final_config['console'],
                                      print_conf_mat=False)
    _, model = train_runner(final_config, model, final_trainer, save_path)

    return model, save_path, final_config


if __name__ == '__main__':
    config = Setup(config)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets_', 'accuracy', 'training time', 'testing time']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"
    datasets = os.listdir(data_path)
    datasets.sort()

    for problem in datasets:  # for loop on the all datasets in "data_dir" directory
        for n_tral in range(100):
            X_train , y_train = load_from_tsfile(data_path+problem+'/TRAIN_default_X.ts')
            X_test , y_test = load_from_tsfile(data_path+problem+'/TEST_default_X.ts')
            train_loader, val_loader, dev_dataset, test_loader = load_data_ConvTran(X_train, X_test, y_train, y_test,
                                                                        to_half=problem=="EigenWorms")
            #----------------------------------------------------------------------------
            # -------------------------------------------- Build Model -----------------------------------------------------
            start_tr = timeit.default_timer()
            model , save_path, final_config = build_train_model(train_loader, val_loader, dev_dataset)
            tot_tr = timeit.default_timer() - start_tr

            start_te = timeit.default_timer()
            model._predict(test_loader)
            tot_te = timeit.default_timer() - start_te
            #best_model, optimizer, start_epoch = load_model(model, save_path, final_config['optimizer'])
            best_model = model
            best_model.to(device)

            best_test_evaluator = SupervisedTrainer(best_model, test_loader, device, final_config['loss_module'],
                                                    print_interval=final_config['print_interval'], console=final_config['console'],
                                                    print_conf_mat=True)
            best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
            print( "best model accuracy  %.2f " % best_aggr_metrics_test['accuracy'] ,
                "\t precision  %.2f " % best_aggr_metrics_test['precision'], "\t loss  %.2f " % best_aggr_metrics_test['loss'])


            All_Results = np.vstack((All_Results, [
                problem, best_aggr_metrics_test['accuracy'] , tot_tr, tot_te ]))

            All_Results_df = pd.DataFrame(All_Results)
            All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))




