import sys
import policy_network_crowd as pl

sent_vec = "../data/sentence_vector.out"
prob = "../data/probabilities.out"

train_data = sys.argv[1]
pretrained_cnn_path = sys.argv[1]
pretrained_policy_path = sys.argv[2]
output_cnn_path = sys.argv[3]
output_policy_path = sys.argv[4]
pl.startRL(pretrained_cnn_path,pretrained_policy_path, output_cnn_path, output_policy_path,
	train_data_,sen_tvec,prob)