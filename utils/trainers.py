from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import *
from models.MyMiniRocket import MyMiniRocket
from utils.data_utils import load_data_ConvTran
from models.convTran import build_train_ConvTran

def trainScore_hydra_gpu( dataset , device, batch_size ):

    # get different dataset parts
    X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
    X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

    data_train = Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    data_test = Dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    # extract TS info
    _ , n_channels, length = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    transform = HydraMultivariateGPU(input_length=length, num_channels=n_channels).to(device)
    model = RidgeClassifier(transform=transform, device=device)
    model.fit(data_train, num_classes=n_classes)

    error_test_set  =   model.score(data_test)

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

    return acc_test_set.item(), model



def train_ConvTran( dataset , device, batch_size, verbose=False ):

    train_loader, val_loader, dev_dataset,test_loader = load_data_ConvTran(
        dataset, batch_size=batch_size)

    convTran, hyperParams = build_train_ConvTran(train_loader, val_loader, dev_dataset, device=device,
                                                 save_path=None,verbose=verbose)
    convTran.eval()

    accuracy_testSet = convTran.score(test_loader)

    return accuracy_testSet.item(), convTran


trainer_list = [
    ('hydra' ,        trainScore_hydra_gpu      , 128 ),
    #('miniRocket',   train_Minirocket_ridge_GPU , 64),
    #('ConvTran',     train_ConvTran             ,32),
]
