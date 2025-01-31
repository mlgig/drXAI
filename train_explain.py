import timeit
from threading import Thread, Event

from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu
from utils.data_utils import *
from explanations import tsCaptum_explainations, windowSHAP_explanations
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

for dir_n in [ "others_new"]:#,,  "Multivariate_ts", "others_MTSCcom"]:
	for current_dataset in ['MP50']: #sorted(os.listdir(os.path.join(bast_path,dir))): # datasets:

		print(current_dataset,":")
		dataset_dir =  os.path.join(base_path,dir_n,current_dataset)
		data = load_datasets(dataset_dir, current_dataset)

		# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
		results[current_dataset] = {'labels_map' : data['labels_map']}

		#################################### train model #################################################

		# set which models to use and initialize result
		model_trainer_batchSize = [
			('ConvTran', train_ConvTran,32),
			('miniRocket', train_Minirocket_ridge_GPU,64) ,
			('hydra', trainScore_hydra_gpu,128),
		]
		results[current_dataset] = dict.fromkeys( [name for (name,_,_) in model_trainer_batchSize  ])


		for model_name,  trainer, batch_size in model_trainer_batchSize:

			# check if current combination of dataset and model is a special case
			if (current_dataset,model_name) in special_cases:
				if special_cases[(current_dataset,model_name)]=='skip':
					continue
				else:
					bast_path =  special_cases[(current_dataset,model_name)]

			# otherwise train and save the model along with additional info
			start_tr = timeit.default_timer()
			save_model_path = os.path.join("saved_models", "_".join((current_dataset,model_name))+".pth")
			accuracy_test, X_train_pred, model = trainer( dataset = data ,device = device, batch_size=batch_size )
			print( '\t', model_name , "'s train completed, accuracy was",accuracy_test)

			torch.save(model,save_model_path)

			results[current_dataset][model_name] = {
				"accuracy_test"		:accuracy_test,
				'training_time'		:(timeit.default_timer() -start_tr)
			}

			################################ explain ###########################################
			# get explaining set's features and labels
			X_to_explain , labels = data['train_set']['X'] , data['train_set']['y']

			# define backgrounds to be tested
			backgrounds = [
				('zeros_background', X_to_explain[0:1]*0	),									#zeros background
				('smote_background',smote_avg(X_to_explain,labels)	),
				('prototypes_background' ,class_prototypes_avg(X_to_explain,labels)	),
				('equal_distributed_background',equal_distributed_proba(X_to_explain,X_train_pred)	),
				('42s_background' ,np.ones((	1, X_to_explain.shape[1], X_to_explain.shape[2]))*42	)	# meaningless background
			]

			results[current_dataset][model_name]['results'] = dict.fromkeys(
				[ name for (name,_) in backgrounds]
				 )

			for b_name,background in backgrounds:

				# for each background initialise result dict; then explain
				results[current_dataset][model_name]['results'][b_name] = {}
				current_experiment = results[current_dataset][model_name]['results'][b_name]

				tsCaptum_explainations(current_experiment=current_experiment
						,model=model,  X=X_to_explain, y=labels, batch_size=batch_size,background=background)

				# use threading for windowSHAP
				to_terminate = Event()
				p = Thread(target = windowSHAP_explanations, args = [
					current_experiment, model,X_to_explain,to_terminate, background
				])
				# set a termination flag after have joined current thread for 24 hours (60*60*24 seconds)
				p.start()	; 	p.join(timeout = 60*60*24)	;	to_terminate.set()	; p.join()

				print('\t', model_name, b_name, 'combination computed')
				# dump result data structure on disk
				np.savez_compressed(os.path.join("results","MP50.npz"), results=results)