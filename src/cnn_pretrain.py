import pycnn
import sys

train_data = sys.argv[1]
model_path = sys.argv[2]
epoch = int(sys.argv[3])

pycnn.startTrain(train_data,model_path,epoch)