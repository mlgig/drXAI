import timeit
import argparse

from utils.data_utils import *
from utils.trainers import *
from explanations import windowSHAP_selection, tsCaptum_selection
from utils.backgrounds import class_prototypes_avg, smote_avg

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
	for dir_name in sorted(os.listdir(args.dataset_dir)):
		for current_dataset in sorted(os.listdir(os.path.join(args.dataset_dir,dir_name) ) ):

			results_path = os.path.join(results_dir, "_".join( (current_dataset ,"results") ) )+".npz"
			dataset_dir =  os.path.join(base_path,dir_name,current_dataset)
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
				# get explaining set's features and labels
				X_to_explain , labels = data['train_set']['X'] , data['train_set']['y']

				# define backgrounds to be tested
				backgrounds = [
					('zerosBackground', X_to_explain[0:1]*0	),
					('smoteBackground',smote_avg(X_to_explain,labels)	),
					('prototypesBackground' ,class_prototypes_avg(X_to_explain,labels)	),
				]

				for b_name,background in backgrounds:

					# for each background initialise result dict, then explain
					results[model_name][b_name] = {}

					key_prefix = 'selected_channels_' if channel_selection else 'selected_timePoints_'
					for alg in ['Feature_Ablation' ,'Shapley_Value_Sampling']:
						ch_selections, attribution, exp_time = tsCaptum_selection(
							model=model,X=X_to_explain,y=labels,batch_size=batch_size,background=background,
							explainer_name=alg,channel_selection=channel_selection
						)

						results[model_name][b_name][alg] = {
							key_prefix+'averageFirst' : ch_selections[0],
							key_prefix+'absoluteFirst' : ch_selections[1],
							key_prefix+'intersection' : list(
								set( ch_selections[0]).intersection(set(ch_selections[1]))
							),
							'saliency_map' : attribution,
							'explaining_time' : exp_time
						}

					if not channel_selection:
						ch_selections, attribution, exp_time = windowSHAP_selection(model,X_to_explain,background,
																					channel_selection=channel_selection)

						results[model_name][b_name]['WindowSHAP'] = {
							key_prefix+'averageFirst' : ch_selections[0],
							key_prefix+'absoluteFirst' : ch_selections[1],
							key_prefix+'intersection' : list(
								set( ch_selections[0]).intersection(set(ch_selections[1]))
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
