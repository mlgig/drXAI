import timeit
import torch

from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu
from utils.data_utils import *
from explanations import tsCaptum_explainations, windowSHAP_explanations

# TODO same datatype or numpy array for predict and also for score?
# TODO avoid double conversion back and fort from ndarray to torch.Tensor for aaltd ridge?
# TODO fix different ways to load data
# TODO do I need tensorboard for ConvTran?

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate result data structure
results = {}

####################### load dataset ##################################
bast_path = "./"

for dir in [ "Multivariate_ts/" , "others_new/" ]:
	for current_dataset in sorted(os.listdir(os.path.join(bast_path,dir))): # datasets:

		# load data	# TODO same code for both!
		dataset_dir = ( dir, os.path.join(bast_path,dir,current_dataset) )
		data = load_datasets(dataset_dir, current_dataset)

		# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
		results[current_dataset] = {'labels_map' : data['labels_map']}

		################################ train and explain ####################################################
		for model_name,  trainer, batch_size in [ ('ConvTran', train_ConvTran,32), ('hydra', trainScore_hydra_gpu,128),
						('miniRocket', train_Minirocket_ridge_GPU,64) ] :	#('QUANT', train_QUANT_aaltd2024,128)

			start_tr = timeit.default_timer()

			# train and save model along with additional info about the training
			save_model_path = os.path.join("saved_models", "_".join((current_dataset,model_name))+".pth")
			accuracy_explain , accuracy_test, model = trainer( dataset = data ,device = device, batch_size=batch_size )

			results[current_dataset][model_name] = {
				"accuracy_explain"	:accuracy_explain,
				"accuracy_test"		:accuracy_test,
				'training_time'		:(timeit.default_timer() -start_tr)
			}

			torch.save(model,save_model_path)
			print(model_name , 'train completed: Accuracy on explain set',accuracy_explain, 'and test set' , accuracy_test)

			# get explaining set's features and labels
			X_to_explain , labels = data['explain_set']['X'] , data['explain_set']['y']

			tsCaptum_explainations(current_experiment=results[current_dataset][model_name],model=model,
								   X=X_to_explain, y=labels, batch_size=batch_size)
			windowSHAP_explanations(current_experiment=results[current_dataset][model_name],model=model,
									X_explain=X_to_explain)

			#np.save(os.path.join("results","experiment_results.npy"), results)
			np.savez_compressed(os.path.join("results","experiment_results.npz"), results=results)