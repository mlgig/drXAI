import timeit
import torch
import numpy as np

from torch.cuda import empty_cache as empty_gpu_cache
from torch.utils.data import  DataLoader
import torch.nn as nn

from .tsai.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features

from .aaltd2024.code.ridge import RidgeClassifier
from .aaltd2024.code.utils import Dataset

# TODO should I change module and file name??
class MyMiniRocket(nn.Module):

    def __init__(self, n_channels,seq_len,n_classes,normalise=True,chunk_size=32, device="cpu", verbose=False):
        super(MyMiniRocket, self).__init__()

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.normalise = normalise
        self.chunk_size = chunk_size
        self.device = device
        self.f_mean =None ; self.f_std = None

        self.transformer_model = MiniRocketFeatures(n_channels,seq_len,device=device)
        self.intermediate_dim = 9996

        self.classifier = RidgeClassifier(self.transformer_model,device=device)
        self.to(device)
        self.trained = False

    """
    def _predict (self,X, batch_size=32):
        outs = []
        n = X.shape[0]

        with torch.no_grad():
            for i in range(0, n,  batch_size):
                curr_X = X[i:min(i+batch_size,n)]
                curr_y = self.classifier._predict(curr_X)
                outs.append(curr_y.cpu().detach().numpy())

        return np.concatenate(outs,axis=0) # TODO hardcoded! torch.tensor(outs)
    """

    def train(self,data_train):

        self.classifier.fit(data_train)
        self.trained = True

    def score(self,data_test):
        error = self.classifier.score(data_test)
        acc = 1-error.cpu().numpy()

        return acc

    def predict(self,data):
        y = self.classifier.predict(data)
        return y

    def predict_proba(self,data):
        y = self.classifier.predict_proba(data)
        return y
