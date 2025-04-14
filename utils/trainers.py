from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.quant import QuantClassifier as QuantClassifier_aaltd
from models.aaltd2024.code.utils import *

from models.MyMiniRocket import MyMiniRocket

from utils.data_utils import load_data_ConvTran
from models.convTran import build_train_ConvTran

def trainScore_hydra_gpu( dataset , device, batch_size ):

    # get different dataset parts
    X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
    X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

    # TODO do I need dataset only for score???
    data_train = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    # extract TS info
    _ , n_channels, length = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    transform = HydraMultivariateGPU(input_length=length, num_channels=n_channels).to(device)
    model = RidgeClassifier(transform=transform, device=device)
    model.fit(data_train, num_classes=n_classes)

    error_test_set  =   model.score(data_test)
    X_train_pred = model.predict_proba(data_train)

    #return   (1 - error_test_set.cpu().numpy().item()), X_train_pred, model
    return   (1 - error_test_set.cpu().numpy().item()), model


def train_Minirocket_ridge_GPU(  dataset , device, batch_size ):

    # get different dataset parts
    X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
    X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

    data_train = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    # extract TS info
    n_samples , n_channels , seq_len = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    model = MyMiniRocket(n_channels=n_channels,seq_len=seq_len,n_classes=n_classes, device=device)
    model.train(data_train)

    acc_test_set = model.score(data_test)
    X_train_pred = model.predict_proba(data_train)

    #    return acc_test_set.item(), X_train_pred, model
    return acc_test_set.item(), model



def train_ConvTran( dataset , device, batch_size, verbose=False ):

    train_loader, val_loader, dev_dataset,test_loader = load_data_ConvTran(
        dataset, batch_size=batch_size)

    convTran, hyperParams = build_train_ConvTran(train_loader, val_loader, dev_dataset, device=device,
                                                 save_path=None,verbose=verbose)
    convTran.eval()

    accuracy_testSet = convTran.score(test_loader)
    X_train_pred = convTran.predict_proba(train_loader)

    #return accuracy_testSet.item(), X_train_pred, convTran
    return accuracy_testSet.item(), convTran






trainer_list = [
    ('hydra', trainScore_hydra_gpu),
    ('miniRocket', train_Minirocket_ridge_GPU),
    ('ConvTran', train_ConvTran),
]


batch_sizes = {
    'hydra' :  128,
    'ConvTran' :  32,
    'miniRocket' : 64,
}


############################# handle special cases #################################
special_cases = {
    ('EigenWorms' , 'ConvTran') : 'skip',
    ('PenDigits', 'miniRocket') : 'skip',
    ('MotorImagery', 'ConvTran') : 16 ,
    ('Tiselac', 	'ConvTran') :  4096,
    ('PenDigits', 	'ConvTran') :  1024
}


def ToSkip_batchSize(current_dataset,clf_name):
    # initialize to default parameters
    to_skip = False
    batch_size = batch_sizes[clf_name]

    if (current_dataset,clf_name) in special_cases:
        if special_cases[(current_dataset,clf_name)]=='skip':
            to_skip = True
        else:
            batch_size = special_cases[(current_dataset,clf_name)]

    return  to_skip, batch_size