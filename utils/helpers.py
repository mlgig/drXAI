import numpy as np
import random
import torch

def extract_timePoints( data, selection):
	new_data = []
	for intervals in sorted(selection):
		start, end = intervals.split(':')
		new_data.append(data [ :,:,int(start):int(end) ] )
	return np.concatenate(new_data,axis=-1)

def set_seed(seed: int = 42):
	"""Set random seeds for reproducibility"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

def get_computed_AI_selections(saliency_map_dict, selection_dict, accuracies, info, channel_sel):

	key2find = 'selected_channels_intersection' if channel_sel else 'selected_timePoints_intersection'
	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue

		if k=='accuracy':
			accuracies[info[1:]] = saliency_map_dict[k]
		elif k==key2find:
			#k_name = k.replace(key2find,'')
			model, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			for model in selection_dict.keys():
				selection_dict[model][explainer] = saliency_map_dict[k]

		elif type(saliency_map_dict[k])==dict :
			get_computed_AI_selections(
				saliency_map_dict[k],selection_dict,accuracies,
				info+"_"+str(k), channel_sel)

	return selection_dict, accuracies
