import numpy as np
import pandas as pd
from copy import deepcopy
import argparse
from pprint import pprint
import timeit

from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu
from utils.data_utils import *

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path =  #TODO argument

# TODO move into separate shared file
special_cases = {
	('EigenWorms' , 'ConvTran') : 'skip',
	('PenDigits', 'miniRocket') : 'skip',
	('MotorImagery', 'ConvTran') : 16 ,
	('Tiselac', 	'ConvTran') :  4096,
	('PenDigits', 	'ConvTran') :  1024
}

def get_AI_selections(saliency_map_dict, result_dict, info):

	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue
		elif k=='selected_channels':
			model, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			result_dict[model][explainer] = saliency_map_dict[k]
		#result_dict[model][explainer+"_cd_selection"] = cd_selection
		elif type(saliency_map_dict[k])==dict :
			get_AI_selections(saliency_map_dict[k],result_dict, info+"_"+str(k))

	return result_dict


def get_elbow_selections(current_data,elbows):
	return {
		'elbow_pairwise' : elbows[current_data]['Pairwise'] ,
		'elbow_sum' : elbows[current_data]['Sum']
	}

def get_accuracies(original_data,save_models_path, channel_selections, initial_accuracies=None):

	# get info
	current_dataset = original_data['name']
	accuracies = {	'accuracies'	: {}	}
	current_dataset_dict = accuracies['accuracies']



	for clf_name, trainer, batch_size in [
		('hydra', trainScore_hydra_gpu, 128),
		('ConvTran', train_ConvTran, 32),
		('miniRocket', train_Minirocket_ridge_GPU, 64),

		]:

		current_dataset_dict[clf_name] = {} if initial_accuracies is None else {
			'initial_accuracy' : initial_accuracies[current_dataset][clf_name]['all_channels']
		}

		for exp_name, selection in channel_selections[clf_name].items():
			# accuracies vector
			current_dataset_accs = np.zeros(shape=(5,))

			# get current selected channels
			data  = deepcopy(original_data)
			data['train_set']['X'] = data['train_set']['X'][:,selection,:]
			data['test_set']['X'] = data['test_set']['X'][:,selection,:]

			#print("current selection is",selection, data['train_set']['X'].shape, data['train_set']['X'].shape)
			# check if current combination of dataset and model is a special case
			if (current_dataset,clf_name) in special_cases:
				if special_cases[(current_dataset,clf_name)]=='skip':
					continue
				else:
					batch_size = special_cases[(current_dataset,clf_name)]

			saved_models_path = os.path.join(save_models_path, "_".join((current_dataset,clf_name,exp_name))+".pth")

			# train 5 times
			for i in range(5):
				star_time = timeit.default_timer()
				current_accuracy, _ , model = trainer(dataset=data, device=device, batch_size=batch_size)
				total_time = timeit.default_timer() - star_time
				current_dataset_accs[i] = current_accuracy

				# TODO optional hyperparam??
				# save best model
				if max(current_dataset_accs)==current_accuracy:
					torch.save(model,saved_models_path)
					training_time = total_time


			# extrac mean, std deviation and best accuracy
			current_dataset_dict[clf_name][exp_name]	 = 	{
				'training_time' : training_time,
				'mean' : np.mean(current_dataset_accs).item(),
				'std' : np.std(current_dataset_accs).item() ,
				'best' :  np.max(current_dataset_accs).item(),
				'channels' : selection
			}

			#("mean and std accuracy was", 	(np.mean(current_acc) ,np.std(current_acc)), "\n\n")

	return current_dataset_dict

def main(args):

	initial_accuracies_path = args.initial_accuracies_path
	saved_models_path = args.saved_models_path
	result_path = args.result_path
	elbow_selections_path = args.elbow_selections_path
	saliency_maps_path = args.saliency_maps_path
	get_initial_accuracy = (initial_accuracies_path==None)

	if  get_initial_accuracy:
		# if currently interested in initial classifier accuracy no need to load any data
		initial_accuracies = None
	else:
		# otherwise load elbow selection, saliency maps and initial accuracies
		print("loading data...", end="\t")
		saliency_maps = np.load(saliency_maps_path,allow_pickle=True)['results'].item()
		all_elbow_selections = np.load(elbow_selections_path, allow_pickle=True).item()
		initial_accuracies = np.load(initial_accuracies_path, allow_pickle=True).item()
		print("done!")

	#TODO both for are hard coded!
	all_accuracies = {}
	for dir in  ["Multivariate_ts/", "others_new/","others_MTSCcom"]: #, "data_new/" ]:	#TODO hardcoded "Multivariate_ts/",
		for current_dataset in sorted(os.listdir(os.path.join(base_path,dir))): # #TODO to remove :2!

			# load current dataset
			dataset_dir =  os.path.join(base_path,dir,current_dataset)
			data = load_datasets(dataset_dir, current_dataset)
			print(data["train_set"]["X"].shape)

			if not get_initial_accuracy:
				# if not initial accuracy mode get elbow and AI selection
				elbow_selections = get_elbow_selections(current_dataset,all_elbow_selections)
				all_selections = get_AI_selections(saliency_maps[current_dataset],{
					'ConvTran' : elbow_selections, 'miniRocket': elbow_selections, 'hydra' : elbow_selections
				},"")

			else:
				# otherwise compute the accuracy using all available channel
				all_channels = {'all_channels' :[i for i in range( data['train_set']['X'].shape[1] )]}
				all_selections = {
					'ConvTran':all_channels,
					'miniRocket':all_channels,
					'hydra': all_channels
				}

			# train models on selected dataset versions
			current_accuracies = get_accuracies(data,saved_models_path, all_selections, initial_accuracies)
			pprint(current_accuracies ,indent=4)
			all_accuracies[current_dataset] = current_accuracies

			np.save( result_path,all_accuracies)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("result_path", type=str, help="file path where final accuracies will be saved")
	parser.add_argument("saved_models_path", type=str, help="folder where trained model will be saved")
	parser.add_argument("initial_accuracies_path", type=str, nargs="?", help="file path where initial accuracies "
															 "will be saved")
	parser.add_argument("elbow_selections_path", type=str, nargs='?', help="file path where elbow selections"
																	 " are saved")
	parser.add_argument("saliency_maps_path", type=str, nargs='?', help="file path where saliency maps and "
							"relative selections are saved. Blank for initial accuracies")	#TODO to be tested
	args = parser.parse_args()
	main(args)