from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import numpy as np
from .ConvTran.utils import dataset_class


def load_data_ConvTran(X_train, X_test, y_train, y_test, val_ratio=0.25, batch_size=16):

	le = LabelEncoder()
	y_train = le.fit_transform(y_train)
	y_test = le.transform(y_test)

	# assuming equal length data
	_ , n_channels , seq_len = X_train.shape

	train_data, train_label, val_data, val_label = split_dataset(X_train, y_train,val_ratio)

	# creating loaders
	train_dataset = dataset_class(train_data, train_label)
	val_dataset = dataset_class(val_data, val_label)
	dev_dataset = dataset_class( X_train, y_train)
	test_dataset = dataset_class(X_test,y_test)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

	return train_loader, val_loader, dev_dataset, test_loader

def split_dataset(data, label, validation_ratio):
	splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio) #, random_state=1234)
	train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
	train_data = data[train_indices]
	train_label = label[train_indices]
	val_data = data[val_indices]
	val_label = label[val_indices]
	return train_data, train_label, val_data, val_label

