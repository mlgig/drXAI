import timeit
from threading import Thread, Event

from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu
from utils.data_utils import *
from explanations import tsCaptum_explainations, windowSHAP_explanations, tsCaptum_selection
from utils.backgrounds import class_prototypes_avg, smote_avg, equal_distributed_proba

# TODO same datatype or numpy array for predict and also for score?
# TODO avoid double conversion back and fort from ndarray to torch.Tensor for aaltd ridge?
# TODO do I need tensorboard for ConvTran?

special_cases = {
	('EigenWorms' , 'ConvTran') : 'skip',
	('PenDigit', 'miniRocket') : 'skip',
	('MotorImagery', 'ConvTran') : 16
}

# get device and instantiate result data structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

####################### load dataset ##################################


base_path = "/media/davide/DATA/datasets/Multivariate2018_ts/"

for dir_n in [ "Multivariate_ts", "others_new", "others_MTSCcom"]:
	for current_dataset in sorted(os.listdir(os.path.join(base_path,dir_n))): # datasets:

		dataset_dir =  os.path.join(base_path,dir_n,current_dataset)
		data = load_datasets(dataset_dir, current_dataset)

		# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
		results[current_dataset] = {'labels_map' : data['labels_map']}

		#################################### train model #################################################

		# set which models to use and initialize result data structure
		models_batchSizes = [
			('hydra',128),

			('ConvTran',32),
			('miniRocket',64) ,
		]

		results[current_dataset] = dict.fromkeys( [name for (name,_) in models_batchSizes  ])

		for model_name, batch_size in models_batchSizes:

			# check if current combination of dataset and model is a special case
			if (current_dataset,model_name) in special_cases:
				if special_cases[(current_dataset,model_name)]=='skip':
					continue
				else:
					bast_path =  special_cases[(current_dataset,model_name)]

			# otherwise load current model
			save_model_path = os.path.join("saved_models", "_".join((current_dataset,model_name))+".pth")
			model = torch.load(save_model_path)

			################################ explain ###########################################
			# get explaining set's features and labels
			X_to_explain , labels = data['train_set']['X'] , data['train_set']['y']

			# define backgrounds to be tested
			backgrounds = [
				#('zeros_background', X_to_explain[0:1]*0	),									#zeros background
				('smote_background',smote_avg(X_to_explain,labels)	),
				('prototypes_background' ,class_prototypes_avg(X_to_explain,labels)	),
				#('equal_distributed_background',equal_distributed_proba(X_to_explain,X_train_pred)	),
				#('42s_background' ,np.ones((	1, X_to_explain.shape[1], X_to_explain.shape[2]))*42	)	# meaningless background
			]

			results[current_dataset][model_name] = dict.fromkeys([ name for (name,_) in backgrounds])

			for b_name,background in backgrounds:

				for alg in ['Feature_Ablation' ,'Shapley_Value_Sampling']:
					# for each background initialise result dict; then explain
					chs_selected, attribution = tsCaptum_selection(model=model,X=X_to_explain,y=labels,
						batch_size=batch_size,background=background,explainer_name=alg, return_saliency=True)
					results[current_dataset][model_name][b_name] = {
						'selected_channels' : chs_selected,
						'saliency_map' : attribution
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
				np.savez_compressed(os.path.join("results","saliency_maps.npz"), results=results)

