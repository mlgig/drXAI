import timeit
import argparse

from utils.data_utils import *
from utils.trainers import *
from drXAI import drXAI

def main(args):

	#get arguments
	base_path = args.dataset_dir
	saved_models_dir = args.saved_models_path
	results_dir = args.explainer_results_dir
	random_seed = args.random_seed

	channel_selection =  args.channel_selection
	print("performing channel selection") if channel_selection else print("performing time point selection")

	# get device, set random seed and instantiate result data structure
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(random_seed)

	# load dataset
	for current_dataset in sorted(os.listdir(args.dataset_dir ) ):

		results_path = os.path.join(results_dir, "_".join( (current_dataset ,"results") ) )+".npz"
		dataset_dir =  os.path.join(base_path,current_dataset)
		data = load_datasets(dataset_dir, current_dataset)

		print("\n\n current loaded dataset is....", current_dataset)

		# create an entry in result's data structure. Save 'symbolic label -> numeric label' map
		results = {'labels_map' : data['labels_map']}

		############################# train ####################################
		# train current classifier
		for model_name, trainer, batch_size in trainer_list:

			# if not to skip record accuracy and save model to disk
			start_time = timeit.default_timer()
			current_accuracy , model = trainer(dataset=data, device=device, batch_size=batch_size)
			training_time = timeit.default_timer() - start_time

			file_name = "_".join((current_dataset,model_name,"allChannel"))+".pth"
			torch.save(model, os.path.join(saved_models_dir,file_name))

			results[model_name] = {
				"training_time" : training_time,
				'accuracy' : current_accuracy
			}

			################################ explain ###########################################

			# TODO clean from here to the bottom!
			# TODO add comments all over the script(s) (drXAI, compute_metrics, etc)

			# get explaining set's features and labels
			X2explain , labels = data['train_set']['X'] , data['train_set']['y']

			# define backgrounds to be tested
			#backgrounds = [
			#	('zerosBackground', X_to_explain[0:1]*0	),
			#	('smoteBackground',smote_avg(X_to_explain,labels)	),
			#	('prototypesBackground' ,class_prototypes_avg(X_to_explain,labels)	),
			#]

			backgrounds2use = ["zeros","SMOTE","Proto"]
			for b_name in backgrounds2use:

				# for each background initialise result dict, then explain
				results[model_name][b_name] = {}

				key_prefix = 'selected_channels_' if channel_selection else 'selected_timePoints_'

				explainers2use = [ "Feature_Ablation", "SHAP"] if channel_selection  else  \
					["Feature_Ablation", "SHAP","WindowSHAP"]
				for e_name in explainers2use:
					drxai = drXAI(channel_selection=channel_selection, classifier=model,dataset_X=X2explain,
								  dataset_y=labels, explainer_name=e_name, background_name=b_name,
								  explainer_kwargs={'batch_size':batch_size})
					selections, attribution,exp_time = drxai.get_selection()


					results[model_name][b_name][e_name] = {
						key_prefix+'averageFirst' : selections[0],
						key_prefix+'absoluteFirst' : selections[1],
						key_prefix+'intersection' : list(
							set( selections[0]).intersection(set(selections[1]))
						),
						'saliency_map' : attribution,
						'explaining_time' : exp_time
					}

					print('\t', model_name, b_name, 'combination computed')

				# dump result data structure on disk
				np.savez_compressed(results_path, results=results)


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


# TODO classifier and batch_size as argument i.e. remove trainer_list from utils.trainers

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str, help="folder where datasets are stored")
	parser.add_argument("saved_models_path", type=str, help="folder where to save models")
	parser.add_argument("explainer_results_dir", type=str, help="directory where to save classifiers and "
				 "attributions info including related selection. Format is one file per dataset")
	parser.add_argument("random_seed", type=int, help="random seed to be used for reproducibility")
	parser.add_argument('--channel_selection',type=str2bool, default=False, help="whether to perform channel selection "
				"or not (perform time point selection)")

	args = parser.parse_args()
	main(args)
