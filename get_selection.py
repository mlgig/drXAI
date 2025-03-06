import timeit
from threading import Thread, Event
import argparse

import numpy as np

from utils.trainers import trainer_list
from utils.data_utils import *
from utils.batchSizes_exceptions import batch_sizes as batches_dict, ToSkip_batchSize
from explanations import windowSHAP_explanations, tsCaptum_selection
from utils.backgrounds import class_prototypes_avg, smote_avg, equal_distributed_proba

# TODO same datatype or numpy array for predict and also for score?
# TODO avoid double conversion back and fort from ndarray to torch.Tensor for aaltd ridge?
# TODO do I need tensorboard for ConvTran?

def main(args):

	#get arguments
	base_path = args.dataset_dir
	saved_models_dir = args.saved_models_path
	model_result_path = args.model_result_path
	results_path = args.explainer_results_path
	random_seed = args.random_seed

	# get device, set random seed and instantiate result data structure
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(random_seed)
	results = {}
	models_info = {}

	# load dataset
	# TODO hard coded
	for dir_n in [ "Multivariate_ts", "others_new", "others_MTSCcom"]:
		for current_dataset in  sorted(os.listdir(os.path.join(base_path,dir_n))): # datasets:

			dataset_dir =  os.path.join(base_path,dir_n,current_dataset)
			data = load_datasets(dataset_dir, current_dataset)

			print("\n\n current loaded dataset is", current_dataset)

			# TODO hard coded
			if data['train_set']['X'].shape[1] < 8:
				continue

			# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
			results[current_dataset] = {'labels_map' : data['labels_map']}
			models_info[current_dataset] = {}

			############################# train ####################################
			# train current classifier
			for model_name, trainer in trainer_list:

				# check whether to skip or not current classifier/dataset combination
				to_skip, batch_size = ToSkip_batchSize(current_dataset, model_name)
				if to_skip:
					continue
				else:
					# if not to skip record accuracy and save model to disk
					start_time = timeit.default_timer()
					current_accuracy , model = trainer(dataset=data, device=device, batch_size=batch_size)
					training_time = timeit.default_timer() - start_time

					file_name = "_".join((current_dataset,model_name,"allChannel"))+".pth"
					torch.save(model, os.path.join(saved_models_dir,file_name))
					models_info[current_dataset][model_name] = {
						"training_time" : training_time,
						'accuracy' : current_accuracy,
					}


				results[current_dataset][model_name] = {}

				################################ explain ###########################################
				# get explaining set's features and labels
				X_to_explain , labels = data['train_set']['X'] , data['train_set']['y']

				# define backgrounds to be tested
				backgrounds = [
					('zerosBackground', X_to_explain[0:1]*0	),									#zeros background
					('smoteBackground',smote_avg(X_to_explain,labels)	),
					('prototypesBackground' ,class_prototypes_avg(X_to_explain,labels)	),
				]

				for b_name,background in backgrounds:

					# for each background initialise result dict, then explain
					results[current_dataset][model_name][b_name] = {}

					for alg in ['Feature_Ablation' ,'Shapley_Value_Sampling']:
						ch_selections, attribution, exp_time = tsCaptum_selection(model=model,X=X_to_explain,y=labels,
							batch_size=batch_size,background=background,explainer_name=alg, return_saliency=True)

						# save saliency map, selections,
						results[current_dataset][model_name][b_name][alg] = {
							'selected_channels_absolute' : ch_selections[0],
							'selected_channels_PosNeg' : ch_selections[1],
							'saliency_map' : attribution,
							'explaining_time' : exp_time
						}

						# use threading for windowSHAP
					#to_terminate = Event()
					#p = Thread(target = windowSHAP_explanations, args = [
					#	current_experiment, model,X_to_explain,to_terminate, background
					#])
					# set a termination flag after have joined current thread for 24 hours (60*60*24 seconds)
					#p.start()	; 	p.join(timeout = 60*60*24)	;	to_terminate.set()	; p.join()

					print('\t', model_name, b_name, 'combination computed')

					# dump result data structure on disk
					np.savez_compressed(results_path, results=results)
					np.save( model_result_path, models_info)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str, help="folder where datasets are stored")
	parser.add_argument("saved_models_path", type=str, help="folder where to saved models")
	parser.add_argument("model_result_path", type=str, help="path where to save info about model")
	parser.add_argument("explainer_results_path", type=str, help="path where to save explanations results"
																 "i.e. saliency map and selected channels/time points")
	parser.add_argument("random_seed", type=int, help="random seed to be used for reproducibility")
	args = parser.parse_args()
	main(args)