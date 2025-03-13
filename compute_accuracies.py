import argparse
from copy import deepcopy
from pprint import pprint

from utils.data_utils import *
from utils.get_accuracy import get_accuracies
from explanations import get_AI_selections, get_elbow_selections ,  add_mostAccurate


def main(args):

	explanation_dir = args.explanation_dir
	saved_models_path = args.saved_models_path
	dataset_base_path = args.dataset_dir
	result_path = args.result_path
	elbow_selections_path = args.elbow_selections_path

	# otherwise load elbow selection, saliency maps and initial accuracies
	all_elbow_selections = np.load(elbow_selections_path, allow_pickle=True).item()

	#TODO both for are hard coded!
	all_accuracies = {}
	for dir in  [ "Multivariate_ts" ,"others_new/","others_MTSCcom"]: #, "data_new/" ]:	#TODO hardcoded "Multivariate_ts/",
		for current_dataset in sorted(os.listdir(os.path.join(dataset_base_path,dir))): # #TODO to remove :2!

			# load current dataset
			print("loading ", current_dataset,"...",end="\t")
			dataset_dir =  os.path.join(dataset_base_path,dir,current_dataset)
			data = load_datasets(dataset_dir, current_dataset)
			if (data["train_set"]["X"].shape[1]<8):
				print("skipping cause it's too small")
				continue
			else:
				print("loading attribution as well...", end="\t")
				XAI_results = np.load(os.path.join(explanation_dir, current_dataset+"_results.npz"),
									  allow_pickle=True)['results'].item()
				print("loaded!\n")

			# get elbow selections and AI ones
			elbow_selections = get_elbow_selections(current_dataset,all_elbow_selections)
			all_selections, init_accuracies = get_AI_selections(XAI_results,{
				'ConvTran' : deepcopy(elbow_selections),
				'miniRocket': deepcopy(elbow_selections),
				'hydra' : deepcopy(elbow_selections)
			},{},"")

			all_selections = add_mostAccurate(all_selections,init_accuracies)

			# train models on selected dataset versions
			current_accuracies = get_accuracies(data,saved_models_path, all_selections, init_accuracies)
			pprint(current_accuracies ,indent=4)
			all_accuracies[current_dataset] = current_accuracies

			np.save( result_path ,all_accuracies)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("explanation_dir", type=str, help="dir where explanation results are stored")
	parser.add_argument("saved_models_path", type=str, help="folder where models will be saved")
	parser.add_argument("dataset_dir", type=str, help="directory where datasets are located.")
	parser.add_argument("result_path", type=str, help="path where to store new accuracies")
	parser.add_argument("elbow_selections_path", type=str, nargs="?", help="file path where elbow selections"
																	 " are saved")
	args = parser.parse_args()
	main(args)