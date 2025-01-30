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

for dir in [ "others_new"]:#,,  "Multivariate_ts", "others_MTSCcom"]:
	for current_dataset in ['MP50']: #sorted(os.listdir(os.path.join(bast_path,dir))): # datasets:

		print(current_dataset)
		# load data	# TODO same code for both!
		dataset_dir =  os.path.join(bast_path,dir,current_dataset)
		data = load_datasets(dataset_dir, current_dataset)

		# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
		results[current_dataset] = {'labels_map' : data['labels_map']}

		#################################### train model #################################################

		for model_name,  trainer, batch_size in [('ConvTran', train_ConvTran,32),
			('miniRocket', train_Minirocket_ridge_GPU,64) ,('hydra', trainScore_hydra_gpu,128)
			] :	#('QUANT', train_QUANT_aaltd2024,128)

			# check if current combination of dataset and model is a special case
			if (current_dataset,model_name) in special_cases:
				if special_cases[(current_dataset,model_name)]=='skip':
					continue
				else:
					bast_path =  special_cases[(current_dataset,model_name)]


			start_tr = timeit.default_timer()

			# train and save model along with additional info about the training
			save_model_path = os.path.join("saved_models", "_".join((current_dataset,model_name))+".pth")
			accuracy_test, X_train_pred, model = trainer( dataset = data ,device = device, batch_size=batch_size )
			print(model_name , "'s train completed, accuracy was",accuracy_test)

			torch.save(model,save_model_path)

			results[current_dataset][model_name] = {
				"accuracy_test"		:accuracy_test,
				'training_time'		:(timeit.default_timer() -start_tr)
			}

			################################ explain ###########################################
			# get explaining set's features and labels
			X_to_explain , labels = data['train_set']['X'] , data['train_set']['y']

			backgrounds = [
				X_to_explain[0:1]*0,									#zeros background
				smote_avg(X_to_explain,labels),
				class_prototypes_avg(X_to_explain,labels),
				equal_distributed_proba(X_to_explain,X_train_pred),
			]


			for background in backgrounds:

				tsCaptum_explainations(current_experiment=results[current_dataset][model_name],model=model,
									   X=X_to_explain, y=labels, batch_size=batch_size,background=background)

				# use trheading for windowSHAP
				to_terminate = Event()
				p = Thread(target = windowSHAP_explanations, args = [
					results[current_dataset][model_name], model,X_to_explain,to_terminate, background
				])
				# set a termination flag after have joined current thread for 24 hours (60*60*24 seconds)
				p.start()	; 	p.join(timeout = 60*60*24)	;	to_terminate.set()	; p.join()

			# dump result data structure on disk
			np.savez_compressed(os.path.join("results","MP8.npz"), results=results)