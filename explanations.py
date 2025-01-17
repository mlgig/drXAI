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

def windowSHAP_explanations(current_experiment,model, X_explain):
	# so far hard coded 0 baseline # TODO get more baselines

	background_data = X_explain[0:1]*0

	n_time_points = X_explain.shape[-1]
	w_len =  int(n_time_points/5)	; stride = int(n_time_points/10)

	start_exp = timeit.default_timer()

	explainer = MyWindowSHAP(model.predict_proba, test_data = X_explain, background_data = background_data,
							 							window_len = w_len, stride = stride, method = 'sliding')
	saliency_maps = explainer.shap_values()

	current_experiment['Window_SHAP'] = {
		'window_len' :  w_len,
		'stride' : stride,
		'explaining_time' : timeit.default_timer()- start_exp,
		'results' : saliency_maps
	}

