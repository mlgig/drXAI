import numpy as np
from tsCaptum.explainers import Feature_Ablation, Shapley_Value_Sampling
from windowshap import MyWindowSHAP
import timeit
from utils.channels_extraction import _detect_knee_point


def extract_selection_attribution(attribution, abs):
	"""
	function to extract relevant time points/channels out of a saliency maps
	:param attribution:		the saliency map where to extract relevant attribution
	:param abs: 			whether to use the absolut value i.e. one knee_cut or not i.e. two different knee_cuts
	:return: 				selected channels
	"""
	chs_relevance = np.average(np.average(attribution , axis=-1),axis=0)

	ordered_scores_idx = lambda x : ( np.flip(np.sort(x)) , np.flip(np.argsort(x)) )

	if abs:
		# have only on knee cut based on the channel relevance score's absolute value
		chs_relevance =  np.abs(chs_relevance)
		ordered_relevance, ordered_idx = ordered_scores_idx(chs_relevance)
		knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )
	else:
		# otherwise compute two difference ones and take their union
		ordered_relevance, ordered_idx = ordered_scores_idx(chs_relevance)
		negatives = np.where(ordered_relevance<0)[0]
		if negatives.shape==(0,):
			# no negative values
			knee_selection = []
		else:
			negative_start_idx = negatives[0]

			pos_knee_selection = _detect_knee_point(ordered_relevance[:negative_start_idx],ordered_idx[:negative_start_idx] )
			neg_knee_selection = _detect_knee_point(ordered_relevance[negative_start_idx:],ordered_idx[negative_start_idx:] )
			print( "pos:",len(pos_knee_selection),"out of", len(ordered_relevance[:negative_start_idx]) ,
				   "neg:",len(neg_knee_selection),"out of", len(ordered_relevance[negative_start_idx:]))
			knee_selection = pos_knee_selection + neg_knee_selection

	return knee_selection

###################################	explainers #############################################

def windowSHAP_explanations(current_experiment,model, X_explain, to_terminate,background_data):
	# so far hard coded 0 baseline # TODO get more baselines
	n_instances, n_channels ,n_time_points = X_explain.shape
	w_len =  int(n_time_points/5)	; stride = max(int(n_time_points/10),1)
	print(w_len,stride)

	current_experiment['Window_SHAP'] = {
		'window_len' :  w_len,
		'stride' : stride,
	}

	start_exp = timeit.default_timer()

	saliency_maps = []
	# explain instance by instance
	for i in range(X_explain.shape[0]):
		current_saliency_map = MyWindowSHAP(model.predict_proba, test_data = X_explain[i:i+1],
				background_data = background_data,window_len = w_len, stride = stride, method = 'sliding').shap_values()
		saliency_maps.append(current_saliency_map)
		# is termination flag has been set by the main thread break the loo[
		if to_terminate.is_set():
			current_experiment['Window_SHAP']['timeout'] = True
			print("WindowSHAP has run out of time")
			break

	# before exiting from function save results and elapsed time
	current_experiment['Window_SHAP']['results'] = np.concatenate(saliency_maps)
	current_experiment['Window_SHAP']['explaining_time'] = (timeit.default_timer()- start_exp)


def tsCaptum_selection(model, X, y, batch_size,background, explainer_name, return_saliency):

	# check explainer to be used
	if explainer_name=='Feature_Ablation':
		algo = Feature_Ablation
	elif explainer_name=='Shapley_Value_Sampling':
		algo = Shapley_Value_Sampling
	else:
		raise ValueError("only Feature_Ablation and Shapley_Value_Sampling are allowed")

	explainer = algo(model)
	#TODO n_segment is hard coded
	start_time = timeit.default_timer()
	saliency_map = explainer.explain(samples=X, labels=y, n_segments=1,normalise=True,
											baseline=background,batch_size=batch_size)
	tot_time = timeit.default_timer() - start_time

	selections = []
	for absolute in [True, False]:
		selection = extract_selection_attribution(saliency_map, abs=absolute)
		selections.append( selection )

	# return accordingly to parameters
	to_return = (selections, saliency_map,tot_time) if return_saliency else (selections,tot_time)

	return to_return



