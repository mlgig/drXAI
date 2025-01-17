from aeon.datasets import load_from_ts_file
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
import numpy as np
from models.ConvTran.utils import dataset_class


def load_datasets(dataset_dir_info, current_dataset ):

	# get current dir's info # TODO might be deleted in the future
	dir, dataset_dir = dataset_dir_info

	# data structure for dataset
	data = {
		'train_set' : {},
		'explain_set' : {},
		'test_set' : {}
	}

	# load data from hard drive
	# TODO might be not necessary in the future
	if dir == "Multivariate_ts/":
		X_train, y_train = load_from_ts_file(os.path.join(dataset_dir, current_dataset + "_TRAIN.ts"))
		X_test, y_test = load_from_ts_file(os.path.join(dataset_dir, current_dataset + "_TEST.ts"))
	elif dir == "others_new/":
		X_train, y_train = load_from_ts_file(os.path.join(dataset_dir, "TRAIN_default_X.ts"))
		X_test, y_test = load_from_ts_file(os.path.join(dataset_dir, "TEST_default_X.ts"))
	else:
		raise ValueError("dir not recognized")

	# split train into train and 'explaining set'
	X_train, y_train, train_indices, X_explain, y_explain, val_indices = split_dataset(X_train,y_train,
																					   validation_ratio=0.2,random_state=42)
	data['train_set']['indices'] = train_indices	; data['explain_set']['indices'] = val_indices
	data['train_set']['X'] = X_train;	data['explain_set']['X'] = X_explain;	data['test_set']['X'] = X_test

	# convert to numeric labels
	y_train, y_test, y_explain,labels_map = to_numeric_labels(y_train, y_test, y_explain)
	data['train_set']['y'] = y_train;	data['explain_set']['y'] = y_explain;	data['test_set']['y'] = y_test
	data['labels_map'] = labels_map

	print("loaded dataset",current_dataset, "train set is split into",X_train.shape[0] ," as training set," ,
		  X_explain.shape[0], "as 'explain set'", X_explain.shape[0],". test's dimension are " ,X_test.shape  )
	return data


def load_data_ConvTran(dataset , val_ratio=0.25, batch_size=16):

	# get different dataset parts
	X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
	X_explain, y_explain =  dataset['explain_set']['X'] , dataset['explain_set']['y']
	X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

	# assuming equal length data
	_ , n_channels , seq_len = X_train.shape

	train_data, train_label, _, val_data, val_label, _ = split_dataset(X_train, y_train,val_ratio)

	# creating loaders
	train_dataset = dataset_class(train_data, train_label)
	val_dataset = dataset_class(val_data, val_label)
	explain_dataset = dataset_class(X_explain, y_explain)
	dev_dataset = dataset_class( X_train, y_train)
	test_dataset = dataset_class(X_test,y_test)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	explain_loader = DataLoader(dataset=explain_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

	return train_loader, val_loader, dev_dataset, explain_loader, test_loader




def split_dataset(data, label, validation_ratio, random_state = None):
	splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state) #, random_state=1234)
	train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))

	train_data = data[train_indices]
	train_label = label[train_indices]
	val_data = data[val_indices]
	val_label = label[val_indices]

	return train_data, train_label, train_indices[0] , val_data, val_label, val_indices[0]


def to_numeric_labels(y_train, y_test, y_explain ):

	# convert labels to idx
	le = LabelEncoder()
	y_train = le.fit_transform( y_train)
	y_test = le.transform(y_test)	;	y_explain = le.transform(y_explain)

	return  y_train, y_test, y_explain, le.classes_