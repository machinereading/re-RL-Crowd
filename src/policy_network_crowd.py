import pycnn
import readFile
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import random
import numpy as np
from datetime import datetime
import os

class InstanceSelector:
    def __init__(self,sess,dim_sentVec,dim_entityVec,alpha,net_name="policy_main"):
        self.net_name = net_name
        self.sess = sess
        self.input_dimension = 2*dim_sentVec+2*dim_entityVec
        self.learning_rate = alpha
        self.batch_size=15000

    def build_policy_network_RL(self):
        print("--build " + self.net_name)
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, [None, self.input_dimension], name="policy_input")
            self.y = tf.placeholder(tf.float32, [None, 1], name="policy_radnom_input")
            self.reward = tf.placeholder(dtype=tf.float32, name="reward")
            W_p = tf.get_variable("W_p", [self.input_dimension, 1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_p = tf.get_variable("b_p", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

            self.h = tf.sigmoid(tf.matmul(self.X, W_p) + b_p)
            self.a = tf.cast(self.h > 0.5, dtype=tf.float32)
            policy = tf.clip_by_value(self.y*self.h + (1.-self.y)*(1-self.h),1e-10,1.0)

            self.policy_cost = -tf.reduce_sum(tf.log(policy)*self.reward)
            self.policy_train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.policy_cost)
            self.predict = tf.equal(self.a, self.y)
            self.acc = tf.reduce_mean(tf.cast(self.predict, tf.float32))

    def get_action(self,state):
        p = self.sess.run(self.h,feed_dict={self.X:state})[0]
        n = random.random()
        if n<p:
            return 1.0
        else:   return 0.0

    def decide_action(self,state):
        return self.sess.run(self.a,feed_dict={self.X:state})[0][0]

    def policy_update(self,states,selected, reward):
        loss, train, acc = self.sess.run([self.policy_cost, self.policy_train, self.acc], feed_dict={self.X: states, self.y:selected, self.reward: reward})
        return loss, acc

def concat(sent_vec,avg_sent_vec):
    a = np.append(sent_vec,avg_sent_vec)
    return a

def get_avgVec(selected_sentences,sentences):
    avg_vec = np.zeros((230),dtype=float)
    cnt=0
    for i,x in enumerate(selected_sentences):
        if x==[1.0]:
            cnt+=1
            avg_vec+=sentences[i]
    if cnt==0:
        return avg_vec
    return np.divide(avg_vec,cnt)

def copy_network(dest_scope_name, src_scope_name):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_vars, dst_vars in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_vars))
    return op_holder

def preRL(f,sess,cnn,train_policy,target_policy,num_epoch,tot_avg_reward):
    print("\t-- pre-training Instance Selector with batch size")
    batch_size=1000
    total_batch_size = int(f.train_data_size / batch_size) + 1
    tot_relations = f.read_relations(0,f.train_data_size)
    cnn.load_data(0,f.train_data_size)
    for episode in range(num_epoch):
        avg_loss=0
        avg_reward=0
        avg_acc=0
        tot_selected_sentences=0
        batches = random.sample([x for x in range(total_batch_size)],int(total_batch_size/2))
        batches.sort()
        for t in batches:
            if t%100==0: print(t,datetime.now())
            if random.random()<0.5: continue
            s_idx = t*batch_size
            e_idx = min((t+1)*batch_size,f.train_data_size)
            sentIDs = [x for x in range(s_idx,e_idx)]
            sent_vec = cnn.get_sentence_vector(s_idx,sentIDs)
            relations = tot_relations[s_idx:e_idx]
            sentIDs,sent_vec,relations = shuffle(sentIDs,sent_vec,relations)

            states=[]
            selected_sentIDs=[]
            selected_relations=[]
            selected_ = [[0.0] for x in range(len(sent_vec))]
            for i, s in enumerate(sent_vec):
                avg_sent_vec = get_avgVec(selected_,sent_vec)
                state = concat(s,avg_sent_vec)
                states.append(state)
                selected_[i] = [target_policy.decide_action([state])]
                if selected_[i]==[1.0]:
                    selected_sentIDs.append(sentIDs[i])
                    selected_relations.append(relations[i])
            if selected_sentIDs:
                reward = cnn.get_reward_per_sentence(selected_sentIDs,selected_relations)
            else:
                reward = tot_avg_reward

            loss,acc = train_policy.policy_update(states,selected_,reward*100)
            avg_loss+=(loss/len(batches))
            avg_acc+=(acc/len(batches))
            avg_reward+=(reward/len(batches))
            tot_selected_sentences+=len(selected_sentIDs)
        sess.run(copy_network(dest_scope_name="policy_target",src_scope_name="policy_main"))
        if episode==10: save_model(sess,"../model/policy_network/policy_main_training","policy_main")
        print("Episode: {}, avg_loss: {:.6f}, avg_acc: {:.6f}, avg_reward: {:.6f}, selected sentences: {}".format(episode,avg_loss,avg_acc,avg_reward,tot_selected_sentences))

def RL(sess,f,train_cnn,target_cnn,train_policy,target_policy,num_episode, tot_avg_reward):
    print("\t-- start Reinforcement Learning with batch size")
    batch_size=1000
    total_batch_size = int(f.train_data_size/batch_size)+1
    for episode in range(num_episode):
        avg_loss=0
        avg_acc=0
        avg_reward=0
        tot_selected_sentences=0
        tot_selected_sent_Ids=[]
        target_cnn.load_data(0,f.train_data_size)
        tot_relations = f.read_relations(0,f.train_data_size)
        batches = random.sample([x for x in range(total_batch_size)],int(total_batch_size/2))
        batches.sort()
        for t in batches:
            if t%100==0: print("batch: ", t)
            s_idx = t * batch_size
            e_idx = min((t + 1) * batch_size, f.train_data_size)
            sentIDs = [x for x in range(s_idx, e_idx)]
            sent_vec = target_cnn.get_sentence_vector(s_idx, sentIDs)
            relations = tot_relations[s_idx:e_idx]
            sentIDs, sent_vec, relations = shuffle(sentIDs, sent_vec, relations)

            states = []
            selected_sentIDs = []
            selected_relations = []
            selected_ = [[0.0] for x in range(len(sent_vec))]
            for i, s in enumerate(sent_vec):
                avg_sent_vec = get_avgVec(selected_, sent_vec)
                state = concat(s, avg_sent_vec)
                states.append(state)
                selected_[i] = [target_policy.decide_action([state])]
                if selected_[i] == [1.0]:
                    selected_sentIDs.append(sentIDs[i])
                    selected_relations.append(relations[i])
                    tot_selected_sent_Ids.append(sentIDs[i])
            if selected_sentIDs:
                reward = target_cnn.get_reward_per_sentence(selected_sentIDs, selected_relations)
            else:
                reward = tot_avg_reward

            loss, acc = train_policy.policy_update(states, selected_, reward*100)
            avg_loss += loss / len(batches)
            avg_acc += acc / len(batches)
            avg_reward += reward / len(batches)
            tot_selected_sentences += len(selected_sentIDs)
        print("\t[END] e{} sentence selector training".format(episode),datetime.now())
        if tot_selected_sent_Ids:
            train_cnn.update(tot_selected_sent_Ids,f)
        print("\t[END] e{} update CNN".format(episode),datetime.now())
        sess.run(copy_network(dest_scope_name="cnn_target",src_scope_name="cnn_main"))
        sess.run(copy_network(dest_scope_name="policy_target", src_scope_name="policy_main"))
        print("\t[END] e{} copy network".format(episode),datetime.now())
        target_cnn.save_file(f)
        tot_avg_reward = target_cnn.avg_tot_reward(f.all_labels)
        print("Episode: {}, avg_loss: {:.6f}, avg_reward: {:.6f}, total_avg_reward: {:.6f}, selected sentences: {}".format(episode,avg_loss,avg_reward,tot_avg_reward,len(tot_selected_sent_Ids)))
        print(datetime.now())
        if episode%5==0:
            save_model(sess,"../cnn/cnn_target_training","cnn_target")
            save_model(sess,"../policy_network/policy_network_training","policy_target")

def save_model(sess,model_path,net_name):
    vars_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=net_name)
    saver = tf.train.Saver(vars_model)
    saver.save(sess,model_path)

def startPreRL(train_file,sentence_vec,probabilities,model_path):
    f = readFile.readDS(train_file)
    f.create_bag()
    with tf.Session() as sess:
        c = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_main")
        c.sentence_vector_file = sentence_vec
        c.probabilities_file = probabilities
        avg_reward = c.avg_tot_reward(f.all_labels)
        print("total average reward: {}".format(avg_reward))
        train_policy = InstanceSelector(sess, 230, 0, 0.01, net_name="policy_main")
        target_policy = InstanceSelector(sess, 230, 0, 0.01, net_name="policy_target")
        train_policy.build_policy_network_RL()
        target_policy.build_policy_network_RL()
        sess.run(tf.global_variables_initializer())
        sess.run(copy_network(dest_scope_name="policy_target", src_scope_name="policy_main"))
        preRL(f,sess,c,train_policy,target_policy,50,avg_reward)
        ##To change network name target to main
        sess.run(copy_network(dest_scope_name="policy_main", src_scope_name="policy_target"))
        save_model(sess,model_path,"policy_main")
    print("END")

def startRL(cnn_model,policy_network,target_cnn_model,target_policy_network,train_data,cnn_sentence_vector,cnn_probabilities):
    f = readFile.readDS(train_data)
    f.create_bag()
    tf.logging.set_verbosity(tf.logging.WARN)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session() as sess:
        cnn_main = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_main")
        cnn_target = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_target")
        policy_main = InstanceSelector(sess,230,0,0.01,"policy_main")
        policy_target = InstanceSelector(sess, 230,0,0.01, "policy_target")
        cnn_main.build_model()
        cnn_target.build_model()
        policy_main.build_policy_network_RL()
        policy_target.build_policy_network_RL()
        sess.run(tf.global_variables_initializer())

        print("-- load main cnn & policy models")
        vars_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn_main")
        saver = tf.train.Saver(vars_cnn)
        saver.restore(sess, cnn_model)
        sess.run(copy_network(dest_scope_name="cnn_target", src_scope_name="cnn_main"))
        labels = f.all_labels

        cnn_target.sentence_vector_file = cnn_sentence_vector
        cnn_target.probabilities_file = cnn_probabilities
        avg_reward = cnn_target.avg_tot_reward(labels)
        print("total average reward: {}".format(avg_reward))
        vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_main")
        saver = tf.train.Saver(vars_policy)
        saver.restore(sess, policy_network)
        sess.run(copy_network(dest_scope_name="policy_target", src_scope_name="policy_main"))

        RL(sess, f, cnn_main, cnn_target, policy_main, policy_target, 10, avg_reward)
        print("-- complete training RL")

        vars_RL = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_target")
        saver = tf.train.Saver(vars_RL)
        saver.save(sess, target_policy_network)

        vars_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn_target")
        saver = tf.train.Saver(vars_cnn)
        saver.save(sess, target_cnn_model)

        print("[DONE]")

