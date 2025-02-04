import numpy as np
import pandas as pd
from copy import deepcopy
from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu
from utils.data_utils import *
from pprint import pprint
from utils.channel_extractions import _detect_knee_point, plot_critical_difference

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def extract_selection_attribution(attribution,abs=True):
	if abs:
		attribution = np.abs(attribution)
	chs_relevance_sampeWise = np.average(attribution , axis=-1)

	# selection based on knee point
	chs_relevance_global =  np.average((chs_relevance_sampeWise), axis=0)
	ordered_relevance, ordered_idx = np.flip(np.sort(chs_relevance_global)) , np.flip(np.argsort(chs_relevance_global))
	knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )

	# selection based on critical differences
	cd_selection = plot_critical_difference(
		chs_relevance_sampeWise.transpose((1,0)), [i for i in range(attribution.shape[1])], test="nemenyi"
	)
	return knee_selection



def  get_AI_selections(saliency_map_dict, result_dict, info):

	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue
		elif type(saliency_map_dict[k])==np.ndarray:
			#knee_selection, cd_selection = extract_selection_attribution(saliency_map_dict[k],abs=True)
			knee_selection = extract_selection_attribution(saliency_map_dict[k],abs=True)
			model, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			result_dict[model][explainer+"_knee_selection"] = knee_selection
			#result_dict[model][explainer+"_cd_selection"] = cd_selection
		elif type(saliency_map_dict[k])==dict :
			get_AI_selections(saliency_map_dict[k],result_dict, info+"_"+str(k))

	return result_dict




def get_elbow_selections(current_data,elbows):
	#TODO HARDCODED!
	return {
		'elbow_pairwise' : elbows[current_data]['Pairwise'] ,
		'elbow_sum' : elbows[current_data]['Sum']
	}





def get_accuracies(original_data, channel_selections):

	current_dataset = original_data['name']
	accuracies = {	current_dataset	: {}	}

	for clf_name, trainer, batch_size in [	#('ConvTran', train_ConvTran, 32),
		('miniRocket', train_Minirocket_ridge_GPU, 64),('hydra', trainScore_hydra_gpu, 128)
		]:

		accuracies[current_dataset][clf_name] = {}

		for exp_name, selection in channel_selections[clf_name].items():
			data  = deepcopy(original_data)
			#print("current assessing",exp_name, "originally" , data['train_set']['X'].shape,":")

			current_acc = np.zeros(shape=(5,))
			data['train_set']['X'] = data['train_set']['X'][:,selection,:]
			data['test_set']['X'] = data['test_set']['X'][:,selection,:]

			#print("current selection is",selection, data['train_set']['X'].shape, data['train_set']['X'].shape)

			try:
				for i in range(5):
					_, accuracy_test, _ = trainer(dataset=data, device=device, batch_size=batch_size)
					current_acc[i] = accuracy_test
			except:
				current_acc = np.ones(shape=(5,)) * -1

			accuracies[current_dataset][clf_name][exp_name]	 = 	(np.mean(current_acc) ,np.std(current_acc))
			#("mean and std accuracy was", 	(np.mean(current_acc) ,np.std(current_acc)), "\n\n")

	return accuracies





def main():

	# load elbow selection and saliency maps
	all_elbow_selections = np.load("results/elbow.npy", allow_pickle=True).item()

	#TODO hardcoded! !!!!!!!!!!!!!!!!!!!
	print("loading explanations...", end="\t")
	saliency_maps = np.load("results/MP50.npz",allow_pickle=True)['results'].item()
	print("done!")

	#TODO both for are hard coded!
	for dir in  [  "others_new/"]:# "Multivariate_ts/",  "others_new/"]:# , "data_new/" ]:	#TODO hardcoded
		for current_dataset in ["MP50"]:# sorted(os.listdir(os.path.join(bast_path,dir))): # datasets:

			# load data	# TODO same code for both!
			dataset_dir =  os.path.join(base_path,dir,current_dataset)
			data = load_datasets(dataset_dir, current_dataset)

			elbow_selections = get_elbow_selections(current_dataset,all_elbow_selections)
			all_selections = get_AI_selections(saliency_maps[current_dataset],{
				'ConvTran' : elbow_selections, 'miniRocket': elbow_selections, 'hydra' : elbow_selections
			},"")

			acc = get_accuracies(data,all_selections)
			pprint(acc ,indent=4)


if __name__ == '__main__':
	main()