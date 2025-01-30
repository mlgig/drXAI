import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

def class_prototypes_avg(X_train, y_train):

    num_classes = len(np.unique(y_train))
    prototypes = []
    for i in range(num_classes):
        current_samples = X_train[y_train == i]
        prototypes.append(np.mean(current_samples,axis=0))

    return np.mean(prototypes, axis=0, keepdims=True)

def smote_avg(X_train, y_train):
    n_instance, n_channels, n_time_points = X_train.shape

    sm = SMOTE()
    resampled_X, resampled_y = sm.fit_resample( X_train.reshape((n_instance, -1)),  y_train)

    return np.mean(resampled_X, axis=0).reshape(( 1, n_channels, n_time_points))

def equal_distributed_proba(X_train, X_train_pred):
    idx = np.argmin(np.std(X_train_pred,axis=-1))

    return X_train[idx:idx+1]