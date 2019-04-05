import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from sklearn.utils import shuffle
import sys

print("hello")

class CNN:
    def __init__(self, sess, num_classes, sequence_length, alpha, name="cnn_main"):
        self.word_dim = 100
        self.pos_dim = 5
        self.num_filters = 230
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.learning_rate = alpha
        self.batch_size = 10000
        self.net_name = name
        self.sess = sess
        self.sentence_vector_file = "../data/sentence_vector.out"
        self.probabilities_file = "../data/probabilities.out"

    def build_model(self):
        print("-- build CNN model")
        with tf.variable_scope(self.net_name):
            self.raw_input = tf.placeholder(tf.float32, [None, self.sequence_length, self.word_dim+2*self.pos_dim],
                                            name="raw_input")

            x = tf.expand_dims(self.raw_input, axis=-1)
            self.y = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
            input_dim = x.shape.as_list()[2]

            ##Convolution & Maxpooling layer
            pool_outputs = []
            filter_size = 3
            W_f = tf.get_variable("W_f", [filter_size, input_dim, 1, self.num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_f = tf.get_variable("b_f", [self.num_filters], initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(x, W_f, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.leaky_relu(conv + b_f, name='h')
            max_len = self.sequence_length - filter_size + 1
            pool = tf.nn.max_pool(h, ksize=[1, max_len, 1, 1], strides=[1,1,1,1], padding="VALID")
            pool_outputs.append(pool)

            self.h_pools = tf.reshape(tf.concat(pool_outputs,1), [-1, self.num_filters])

            ##Fully connected layer
            W_r = tf.get_variable("W_r", [self.num_filters, self.num_classes],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_r = tf.get_variable("b_r", [self.num_classes], initializer=tf.constant_initializer(0.1))

            scores = tf.nn.xw_plus_b(self.h_pools, W_r, b_r)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.y))+(tf.nn.l2_loss(W_r)+tf.nn.l2_loss(b_r))*0.0001
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            self.probabilities = tf.nn.softmax(scores)
            self.prediction = tf.argmax(self.probabilities, 1, name="prediction")
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def train(self, num_epoch, file):
        print("Training the CNN model: " + self.net_name)

        for epoch in range(num_epoch):
            total_batch = int(file.train_data_size / self.batch_size)+1
            avg_loss = 0
            avg_acc = 0
            for i in range(total_batch):
                s_idx = i * self.batch_size
                e_idx = min((i + 1) * self.batch_size, file.train_data_size)
                batch_input, labels = file.read_batch_data("train", s_idx, e_idx)
                batch_input,labels = shuffle(batch_input,labels)
                batch_label = tf.one_hot(np.array(labels), self.num_classes).eval()
                feed_dict = {self.raw_input: batch_input,self.y: batch_label}
                acc, l, _ = self.sess.run([self.accuracy, self.loss, self.optimizer], feed_dict=feed_dict)
                avg_loss+=l/total_batch
                avg_acc+=acc/total_batch
                if i%10==0:
                    print("total batch: {}, Processing: {}, Acc:{}, Loss:{}".format(total_batch,i,acc,l))
            # self.save_model("../model/training/cnn_main_1226_ep"+str(epoch),"cnn_main")
            print("Epoch: {}, acc: {}, loss: {}, time: {}".format(epoch, avg_acc, avg_loss,datetime.now()))
            #self.save_model("../model/training/cnn_main_ds44_iter1","cnn_main")


    def test(self,flag,f):
        if flag=="val": size = f.val_data_size
        else: size = f.test_data_size
        input_x, test_labels = f.read_batch_data(flag,0,size)
        input_y = tf.one_hot(np.array(test_labels), self.num_classes).eval()
        acc, prediction = self.sess.run([self.accuracy, self.prediction],
                                                feed_dict={self.raw_input: input_x, self.y: input_y})
        return acc, prediction, test_labels

    def extract(self,f):
        input_x = f.read_extract_data()
        prediction,score = self.sess.run([self.prediction,self.probabilities], feed_dict={self.raw_input:input_x})
        return prediction,score

    def load_data(self,min_id,max_id):
        self.sentVec = pd.read_csv(self.sentence_vector_file, header=None,delim_whitespace=True).values[min_id:max_id]
        return None

    def load_bag_data(self,bag_sentIDs,keys):
        self.sent_vec={}
        sentVec = pd.read_csv(self.sentence_vector_file,header=None,delim_whitespace=True).values
        for key in keys:
            sent_id = bag_sentIDs[key]
            sent_vec = []
            for id in sent_id:  sent_vec.append(sentVec[id])
            self.sent_vec[key] = sent_vec

    def get_sentence_vector(self,st,sentIDs):
        return [self.sentVec[i-st] for i in sentIDs]

    def get_reward(self, sentIDs, label):
        if len(sentIDs)==0:
            return 0.0
        scores = [self.scores[i] for i in sentIDs]
        reward = 0.0
        for s in scores:
            reward += (s[label])
        return reward / len(sentIDs)

    def get_reward_per_sentence(self,sentIDs,labels):
        if len(sentIDs)==0:
            return 0.0

        scores = [self.scores[i] for i in sentIDs]
        reward = 0.0
        for idx,s in enumerate(scores):
            reward+=s[labels[idx]]
        return reward / len(labels)

    def avg_tot_reward(self,labels):
        reward = 0.0
        self.scores = pd.read_csv(self.probabilities_file,header=None,delim_whitespace=True).values
        for idx,s in enumerate(self.scores):
            reward+=(s[labels[idx]])
        return reward/len(labels)

    def update(self, sentIDs,file):
        # sentences,labels = file.read_batch_data_with_id(sentIDs,0,file.train_data_size)
        # input_y = tf.one_hot(np.array(labels),self.num_classes).eval()
        # input_x,input_y = shuffle(sentences,input_y)
        # l,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.raw_input:input_x,self.y:input_y})
        batch_size = 5000
        total_batch = int(len(sentIDs)/batch_size)+1
        sentIDs.sort()
        print("\t--updating cnn with selected sentences, ",len(sentIDs))
        for i in range(total_batch):
            s_idx = i*batch_size
            e_idx = min((i+1)*batch_size,len(sentIDs))
            sentences,labels = file.read_batch_data_with_id(sentIDs[s_idx:e_idx])
            input_y = tf.one_hot(np.array(labels),self.num_classes).eval()
            input_x, input_y = shuffle(sentences,input_y)
            l,_ = self.sess.run([self.loss,self.optimizer],feed_dict={self.raw_input: input_x, self.y:input_y})

    def save_file(self,file):
        if os.path.isfile(self.sentence_vector_file):
            os.remove(self.sentence_vector_file)
        if os.path.isfile(self.probabilities_file):
            os.remove(self.probabilities_file)
        f1 = open(self.sentence_vector_file,'ab')
        f2 = open(self.probabilities_file,'ab')
        batch_size = 30000
        total_batch = int(file.train_data_size/batch_size)+1
        for i in range(total_batch):
            s_idx = i*batch_size
            e_idx = min((i+1)*batch_size,file.train_data_size)
            batch_input,labels = file.read_batch_data("train",st=s_idx,en=e_idx)
            feed_dict={self.raw_input:batch_input}
            sent_vec,scores = self.sess.run([self.h_pools,self.probabilities],feed_dict=feed_dict)
            np.savetxt(f1,sent_vec)
            np.savetxt(f2,scores)
        f1.close()
        f2.close()


    def save_model(self,model,net):
        vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net)
        saver=tf.train.Saver(vars_)
        saver.save(self.sess,model)

	
def startTrain(train_data,model_path,epoch):
    f = readFile.readDS(train_data)
    with tf.Session() as sess:
        c = CNN(sess,f.num_classes,f.sequence_length,0.01,name="cnn_main")
        c.build_model()
        sess.run(tf.global_variables_initializer())
        c.train(epoch)
        c.save_model(model_path,"cnn_main")
        c.save_file(f)