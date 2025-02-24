from utils.trainers import train_ConvTran, train_Minirocket_ridge_GPU, trainScore_hydra_gpu

special_cases = {
	('EigenWorms' , 'ConvTran') : 'skip',
	('PenDigits', 'miniRocket') : 'skip',
	('MotorImagery', 'ConvTran') : 16 ,
	('Tiselac', 	'ConvTran') :  4096,
	('PenDigits', 	'ConvTran') :  1024
}

batch_sizes = {
	'hydra' :  128,
	'ConvTran' :  32,
	'miniRocket' : 64,
}

def ToSkip_batchSize(current_dataset,clf_name):
	# initialize to default parameters
	to_skip = False
	batch_size = batch_sizes[clf_name]

	if (current_dataset,clf_name) in special_cases:
		if special_cases[(current_dataset,clf_name)]=='skip':
			to_skip = True
		else:
			batch_size = special_cases[(current_dataset,clf_name)]

	return  to_skip, batch_size