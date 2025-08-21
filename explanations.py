import numpy as np
import timeit
from windowshap import MyWindowSHAP

from tsCaptum.explainers import Feature_Ablation, Shapley_Value_Sampling
from utils.channels_extraction import _detect_knee_point

####################### functions to extract and order time points selection ##############
def extract_timePoints_features_names(attribution):
	"""
	Extracts feature relevance values and corresponding feature names from the input
	attribution tensor.

	:param attribution: A 3D numpy array representing the saliency map in the format
	(samples, channels, timepoints).

	:return:
	    - feature_relevance: A 1D numpy array containing averaged feature attribution
	     across samples
	    - feature_names: A list of strings each of those indicating start and end
	    position of the corresponding feature.
	"""

	# identify where the unique values of the saliency map are
	intervals = np.where(np.diff(attribution[0])!=0)
	to_gather = np.concatenate( (intervals[1],
								 np.array( [  attribution.shape[-1] -1  ]) 	#add the last element i.e. the last attribution
								 )	)
	# gather these attributions
	all_feature_relevance = np.take_along_axis(  attribution[:,0,:],
												 np.expand_dims(to_gather,axis=0),axis=-1)
	feature_relevance = np.average(all_feature_relevance,axis=0)

	# feature names as startingTimePoint:EndTimePoint
	intervals = ["0"] + to_gather.astype(str).tolist()
	feature_names = [ intervals[i]+":"+intervals[i+1] for i in range( len(intervals)-1 )]

	return feature_relevance, feature_names


def order_timePoints_features_names(feature_relevance, feature_names):
	"""
	Order time points and relative feature names in descending order.

	:param feature_relevance:	vector of features relevance values
	:param feature_names:		vector of corresponding features names

	:return:
		ordered_relevance:		ordered vector of features relevance values
		ordered_idx:			ordered vector of corresponding features names
	"""

	# identify the argsort order
	order = np.flip(np.argsort(feature_relevance))
	# apply the order, i.e. sort, to both feature relevance and feature names
	ordered_relevance = feature_relevance[order].tolist()
	ordered_idx =np.array(feature_names)[order].tolist()

	return ordered_relevance, ordered_idx
	

##################### functions to select features ###############################

def extract_selection_absFirst(attribution, channels=False):
	"""
	function to extract relevant time points/channels out of a saliency maps
	:param attribution:		the saliency map where to extract relevant attribution
	:return: 				selected channels
	"""

	# define extract and sort functions according to time points vs channels selection
	extract_features_names = (
		lambda x : (np.average(np.average(np.abs(x) , axis=-1),axis=0),None)
			) if channels else (
		lambda x :extract_timePoints_features_names(np.abs(x))
	)

	order_features_names = (lambda x,y : ( np.flip(np.sort(x).tolist()) , np.flip(np.argsort(x)).tolist() )) \
		if channels else order_timePoints_features_names	# ignoring the second argument in case of channels selection

	# apply above defined functions
	feature_relevance,feature_names = extract_features_names(attribution)
	ordered_relevance , ordered_idx = order_features_names(feature_relevance,feature_names)

	# knee cut
	knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )

	return knee_selection


def extract_selection_avgFirst(attribution, channels=False):
	"""
	function to extract relevant time points/channels out of a saliency maps
	:param attribution:		the saliency map where to extract relevant attribution
	:return: 				selected channels
	"""

	# define extract and sort functions according to time points vs channels selection
	extract_features_names = (lambda x : (np.average(np.average(x , axis=-1),axis=0),None) ) \
		if channels else extract_timePoints_features_names

	order_features_names = (lambda x,y : ( np.flip(np.sort(x).tolist()) , np.flip(np.argsort(x)).tolist() )) \
		if channels else order_timePoints_features_names

	# apply function to extract feature relevance defined above
	feature_relevance,feature_names = extract_features_names(attribution)
	feature_relevance =  np.abs(feature_relevance)

	ordered_relevance , ordered_idx = order_features_names(feature_relevance, feature_names)
	knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )

	return knee_selection


###################################	explainers #############################################

def windowSHAP_selection(model, X_explain, background_data, channel_selection):

	n_instances, n_channels ,n_time_points = X_explain.shape
	w_len =  np.ceil(n_time_points/6).astype(int).item()	; stride = np.ceil(n_time_points/10).astype(int).item()
	saliency_maps = []

	# explain instance by instance
	start_time = timeit.default_timer()
	for i in range(n_instances):
		print(i, "out of",n_instances,"instances explained")
		current_saliency_map = MyWindowSHAP(model.predict_proba, test_data = X_explain[i:i+1],
				background_data = background_data,window_len = w_len, stride = stride, method = 'sliding').shap_values()
		saliency_maps.append(current_saliency_map)

	tot_time = timeit.default_timer() - start_time

	saliency_maps = np.concatenate( saliency_maps )

	selections = [
		extract_selection_avgFirst(saliency_maps,channels=channel_selection),
		extract_selection_absFirst(saliency_maps,channels=channel_selection)
	]

	return (selections, saliency_maps,tot_time)



def tsCaptum_selection(model, X, y, batch_size,background, explainer_name,channel_selection):

	# check explainer to be used
	if explainer_name=='Feature_Ablation':
		algo = Feature_Ablation
	elif explainer_name=='Shapley_Value_Sampling':
		algo = Shapley_Value_Sampling
	else:
		raise ValueError("only Feature_Ablation and Shapley_Value_Sampling are allowed")

	explainer = algo(model)

	start_time = timeit.default_timer()

	n_segment = 1 if channel_selection else 20
	saliency_map = explainer.explain(samples=X, labels=y, n_segments=n_segment,normalise=False,
											baseline=background,batch_size=batch_size)
	tot_time = timeit.default_timer() - start_time

	selections = [
		extract_selection_avgFirst(saliency_map,channels=channel_selection),
		extract_selection_absFirst(saliency_map,channels=channel_selection)
	]

	return (selections, saliency_map, tot_time)


################################## other functions ##############################################

def get_elbow_selections(current_data,elbows):
	return {
		'elbow_pairwise' : elbows[current_data]['Pairwise'] ,
		'elbow_sum' : elbows[current_data]['Sum']
	}



