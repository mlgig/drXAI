import numpy as np
from tsCaptum.explainers import Feature_Ablation, Shapley_Value_Sampling
from windowshap import MyWindowSHAP
import timeit

def tsCaptum_explainations(current_experiment ,model, X, y, batch_size):
	for (explainer, explainer_name) in [
		(Feature_Ablation, 'Feature_Ablation'), (Shapley_Value_Sampling, 'Shapley_Value_Sampling')
	]:

		current_experiment[explainer_name] = {}
		attribution = explainer(model)
		for n_segments in [1, 5, 10]:
			# TODO think about more baseline!
			start_ex = timeit.default_timer()

			saliency_maps = attribution.explain(samples=X, labels=y, n_segments=n_segments,normalise=True,baseline=0,
											   batch_size=batch_size)
			current_experiment[explainer_name][n_segments] = {
				'result': saliency_maps,
				'explaining_time': (timeit.default_timer() - start_ex)
			}

def windowSHAP_explanations(current_experiment,model, X_explain, to_terminate):
	# so far hard coded 0 baseline # TODO get more baselines
	background_data = X_explain[0:1]*0

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
		current_saliency_map = MyWindowSHAP(model.predict_proba, test_data = X_explain[i:i+1], background_data = background_data,
												window_len = w_len, stride = stride, method = 'sliding').shap_values()
		saliency_maps.append(current_saliency_map)
		# is termination flag has been set by the main thread break the loo[
		if to_terminate.is_set():
			current_experiment['Window_SHAP']['timeout'] = True
			print("WindowSHAP has run out of time")
			break

	# before exiting from function save results and elapsed time
	current_experiment['Window_SHAP']['results'] = np.concatenate(saliency_maps)
	current_experiment['Window_SHAP']['explaining_time'] = (timeit.default_timer()- start_exp)


