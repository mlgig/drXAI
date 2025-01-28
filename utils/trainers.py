import torch

from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.quant import QuantClassifier as QuantClassifier_aaltd
from models.aaltd2024.code.utils import *

from models.MyMiniRocket import MyMiniRocket

from utils.data_utils import load_data_ConvTran
from models.convTran import build_train_ConvTran

def trainScore_hydra_gpu( dataset , device, batch_size ):

    explain_set = 'explain_set' in dataset.keys()

    # get different dataset parts
    X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
    X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

    # TODO do I need dataset only for score???
    data_train = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    if explain_set:
        X_explain, y_explain =  dataset['explain_set']['X'] , dataset['explain_set']['y']
        data_explain = Dataset(X_explain, y_explain, batch_size=batch_size, shuffle=False)

    # extract TS info
    _ , n_channels, length = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    transform = HydraMultivariateGPU(input_length=length, num_channels=n_channels).to(device)
    model = RidgeClassifier(transform=transform, device=device)
    model.fit(data_train, num_classes=n_classes)

    error_explain_set = model.score(data_explain) if explain_set else torch.tensor([2])
    error_test_set  =   model.score(data_test)

    return  (1 - error_explain_set.cpu().numpy().item()) , (1 - error_test_set.cpu().numpy().item()), model



def train_Minirocket_ridge_GPU(  dataset , device, batch_size ):

    explain_set = 'explain_set' in dataset.keys()

    # get different dataset parts
    X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
    X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

    data_train = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    if explain_set:
        X_explain, y_explain =  dataset['explain_set']['X'] , dataset['explain_set']['y']
        data_explain = Dataset(X_explain, y_explain, batch_size=batch_size, shuffle=False)

    # extract TS info
    n_samples , n_channels , seq_len = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    model = MyMiniRocket(n_channels=n_channels,seq_len=seq_len,n_classes=n_classes, device=device)
    model.train(data_train)

    acc_explain_set = model.score(data_explain) if explain_set else np.array([-1])
    acc_test_set = model.score(data_test)

    return acc_explain_set.item(), acc_test_set.item(), model


def train_ConvTran( dataset , device, batch_size, verbose=False ):

    train_loader, val_loader, dev_dataset, explain_loader, test_loader = load_data_ConvTran(dataset, batch_size=batch_size)

    convTran, hyperParams = build_train_ConvTran(train_loader, val_loader, dev_dataset, device=device,
                                                 save_path=None,verbose=verbose)
    convTran.eval()

    accuracy_explainSet =   convTran.score(explain_loader) if explain_loader!=None else np.array([-1])
    accuracy_testSet =      convTran.score(test_loader)

    return accuracy_explainSet.item(), accuracy_testSet.item(), convTran


def train_QUANT_aaltd2024(X_train, y_train, X_test, y_test):
    batch_size = 256

    data_train = BatchDataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = BatchDataset(X_test, y_test, batch_size=batch_size, shuffle=False)
    model = QuantClassifier_aaltd()

    model.fit(data_train)
    error = model.score(data_test)

    return  (1 - error) , model