import numpy as np
import timeit
from windowshap import MyWindowSHAP

from utils.backgrounds import smote_avg, class_prototypes_avg
from tsCaptum.explainers import Feature_Ablation, Shapley_Value_Sampling
from explanations import extract_selection_avgFirst, extract_selection_absFirst

class drXAI:

	def __init__(self,   channel_selection : bool, classifier,dataset_X: np.ndarray, dataset_y : np.ndarray,
				 explainer_name : str , background_name : str , explainer_kwargs):

		# Check argument values
		assert len(dataset_X.shape)==3
		assert len(dataset_y.shape)==1
		assert explainer_name in ["WindowSHAP", "Feature_Ablation", "SHAP"]
		assert background_name in ["zeros","SMOTE","Proto"]

		self.__classifier=classifier
		self.__channel_selection=channel_selection # True if channel_selection , False if time point selection

		self.__X2explain, self.__labels=  dataset_X,dataset_y
		n_instances, n_channels ,n_time_points = self.__X2explain.shape
		#n_classes = len(np.unique(self.__labels))


		# instantiate the requested background
		if background_name=="zeros":
			self.__background= dataset_X[0:1]*0
		elif background_name=="SMOTE":
			self.__background= smote_avg(dataset_X,dataset_y)
		elif background_name=="Proto":
			self.__background= class_prototypes_avg(dataset_X,dataset_y)

		# instantiate the requested explainer
		if explainer_name=="WindowSHAP":
			# WindowSHAP case
			w_len =  explainer_kwargs['w_len'] if 'w_len' in explainer_kwargs \
				else np.ceil(n_time_points/6).astype(int).item()

			stride = explainer_kwargs['stride'] if 'stride' in explainer_kwargs\
				else np.ceil(n_time_points/10).astype(int).item()

			self.__explain_function =MyWindowSHAP(self.__classifier.predict_proba,test_data= self.__X2explain,
					background_data=self.__background,window_len=w_len,stride=stride,method='sliding')		#method arg
						# hardcoded since we only studied the sliding approach

		# Feature Ablation or SHAP case
		elif explainer_name in ["Feature_Ablation", "SHAP"]:
			self.__explain_function = Shapley_Value_Sampling(self.__classifier) if explainer_name=="SHAP" \
				else Feature_Ablation(self.__classifier)

			# for these explainers define kwargs for explanation call
			self.__kwargs = {
				'samples' : self.__X2explain,
				'labels':self.__labels,
				'n_segments': 1 if self.__channel_selection else 20,
				'normalise' : False,
				'baseline' : self.__background,
				'batch_size' : explainer_kwargs['batch_size'] if 'batch_size' in explainer_kwargs else 1
				}


	def get_selection(self,to_return_saliency=True,to_return_time=True):

		# running explanation and measure time
		start_time = timeit.default_timer()
		saliency_maps = self.__explain_function.shap_values()  if isinstance(self.__explain_function,MyWindowSHAP) \
			else self.__explain_function.explain( **self.__kwargs)
		tot_time = timeit.default_timer() - start_time

		# get selections
		selections = [
			extract_selection_avgFirst(saliency_maps,channels=self.__channel_selection),
			extract_selection_absFirst(saliency_maps,channels=self.__channel_selection),
		]

		# add saliency and elapsed time based on optional args
		to_return = [selections]
		if to_return_saliency:
			to_return+=[saliency_maps]
		if to_return_time:
			to_return+=[tot_time]

		return tuple(to_return)
