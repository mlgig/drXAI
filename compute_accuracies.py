import argparse
from copy import deepcopy
from pprint import pprint

from utils.data_utils import *
from utils.get_accuracy import get_accuracies

# get device
base_path =  #TODO argument


def get_AI_selections(saliency_map_dict, result_dict, info):

	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue
		elif k=='selected_channels':
			model, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			result_dict[model][explainer] = saliency_map_dict[k]
		elif type(saliency_map_dict[k])==dict :
			get_AI_selections(saliency_map_dict[k],result_dict, info+"_"+str(k))

	return result_dict


def add_mostAccurate(all_selections,initial_accuracies):
	# get most accurate
	most_accurate_model = max(initial_accuracies, key=lambda model: initial_accuracies[model]['initial_accuracy']['best'])
	AI_selections = ['smote_background_Feature_Ablation', 'smote_background_Shapley_Value_Sampling',
					 'prototypes_background_Feature_Ablation', 'prototypes_background_Shapley_Value_Sampling']
	best_model_AI_selections = [(selection,all_selections[most_accurate_model][selection]) for selection in AI_selections]

	# add this selection to other models
	other_models = set(initial_accuracies.keys()).difference(set([most_accurate_model]))
	for model in other_models:
		for (name,selection) in best_model_AI_selections:
			all_selections[model]["most_accurate_model_"+name] =selection

	return all_selections


def get_elbow_selections(current_data,elbows):
	return {
		'elbow_pairwise' : elbows[current_data]['Pairwise'] ,
		'elbow_sum' : elbows[current_data]['Sum']
	}


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
					'ConvTran' : deepcopy(elbow_selections),
					'miniRocket': deepcopy(elbow_selections),
					'hydra' : deepcopy(elbow_selections)
				},"")
				all_selections = add_mostAccurate(all_selections,initial_accuracies[current_dataset])

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