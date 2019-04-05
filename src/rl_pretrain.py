import sys
import policy_network_crowd as pl

sent_vec = "../data/sentence_vector.out"
prob = "../data/probabilities.out"

train_data = sys.argv[1]
policy_model_path = sys.argv[2]

pl.startPreRL(train_data,sent_vec,prob,policy_model_path)