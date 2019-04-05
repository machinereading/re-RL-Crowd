import make_bag_large
import pycnn
import tensorflow as tf
import random
import numpy as np
from datetime import datetime
import time
import os

class InstanceSelector:
    def __init__(self,file,sess,dim_sentVec,dim_entityVec,alpha,net_name="policy_main"):
        self.net_name = net_name
        self.sess = sess
        self.input_dimension = 2*dim_sentVec+2*dim_entityVec
        self.learning_rate = alpha
        self.build_policy_network()
        self.f = file
        self.batch_size=15000
    def build_policy_network(self):
        print("--build "+self.net_name)
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32,[None,self.input_dimension],name="policy_input")
            self.y = tf.placeholder(tf.float32,[None],name="policy_radnom_input")
            self.reward = tf.placeholder(dtype=tf.float32,name="reward")
            # self.policy = tf.placeholder(dtype=tf.float32,name="policy")
            W_p = tf.get_variable("W_p",[self.input_dimension,1],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_p = tf.get_variable("b_p",[1],dtype=tf.float32,initializer=tf.constant_initializer(0.1))

            self.h = tf.sigmoid(tf.matmul(self.X,W_p)+b_p)
            self.a = tf.cast(self.h>0.5,dtype=tf.float32)
            policy = self.y*tf.clip_by_value(self.h,1e-8,1.) + (1-self.y)*tf.clip_by_value(1-self.h,1e-8,1.)
            self.policy_cost = -tf.reduce_sum(tf.log(policy)*self.reward)
            self.policy_train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.policy_cost)


    def get_action(self,state):
        p = self.sess.run(self.h,feed_dict={self.X:state})[0]
        n = random.random()
        if n<p:
            return 0.0
        else:   return 1.0
    def decide_action(self,state):
        return self.sess.run(self.a,feed_dict={self.X:state})[0][0]
    def egreedy_action(self,state,e):
        e_greedy = 1/(e/5+1)
        if random.random()<e_greedy:
            return float(np.random.randint(2))
        return self.sess.run(self.a,feed_dict={self.X:state})[0][0]
    def policy_update(self,states,selected, reward):
        loss, train = self.sess.run([self.policy_cost, self.policy_train], feed_dict={self.X: states, self.y:selected, self.reward: reward})
        return loss


def concat(sent_vec,avg_sent_vec):
    a = np.append(sent_vec,avg_sent_vec)
    return a

# def concat(sent_vec,avg_sent_vec,e1,e2):
#     a = np.append(sent_vec,avg_sent_vec)
#     b = np.append(a,e1)
#     return np.append(b,e2)

def get_avgVec(selected_sentences,sentences):
    avg_vec = np.zeros((230),dtype=float)
    cnt=0
    for i,x in enumerate(selected_sentences):
        if x==1.0:
            cnt+=1
            avg_vec+=sentences[i]
    if cnt==0:
        return avg_vec
    return np.divide(avg_vec,cnt)

def copy_network(*, dest_scope_name, src_scope_name):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_vars, dst_vars in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_vars))
    return op_holder

def assign_variables(sess, dest_scope_name,src_scope_name,rate):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=dest_scope_name)
    for src_vars,dst_vars in zip(src_vars,dst_vars):
        sess.run(tf.assign(dst_vars,rate*src_vars+(1-rate)*dst_vars))

def preRLwithSen(f,sess,cnn,train_policy,target_policy,num_epoch,avg_reward):
    print("-- pre-training Instance Selector with fixed CNN model and negative sets")
    print("Start time: {}".format(datetime.now()))
    keys = list(f.bag_sentIDs.keys())
    print(f.train_data_size, len(keys))
    best_reward = -99999.9
    for epoch in range(num_epoch):
        random.shuffle(keys)  # shuffling bags
        total_selected_sentences = 0
        total_selected_reward = 0
        total_batch = int(len(keys) / 10000)
        total_avg_loss = 0
        for t in range(total_batch):
            s_idx = t * 10000
            e_idx = min((t + 1) * 10000, len(keys))
            batch_keys = keys[s_idx:e_idx]
            min_id = f.train_data_size + 1
            max_id = 0
            for key in batch_keys:
                for id in f.bag_sentIDs[key]:
                    if id < min_id:   min_id = id
                    if id > max_id:   max_id = id
            cnn.load_data(min_id, max_id)
            for idx, key in enumerate(batch_keys):  # select a bag
                # if idx%100==0:print(idx)
                # vec_E1 = f.bag_e1[key]
                # vec_E2 = f.bag_e2[key]
                label = f.bag_label[key]
                sent_IDs = f.bag_sentIDs[key]
                ne_sent_IDs = f.sampling_sents(key,min_id,max_id)
                sent_vec = cnn.get_sentence_vector(min_id, sent_IDs)
                ne_sent_vec = cnn.get_sentence_vector(min_id,ne_sent_IDs)
                selected_ = [0.0 for x in range(len(sent_vec+ne_sent_vec))]
                selected_sentIDs = []
                states = []
                all_sent_vec = sent_vec+ne_sent_vec
                for i, s in enumerate(all_sent_vec):
                    avg_sent_vec = get_avgVec(selected_, all_sent_vec)
                    state = concat(s, avg_sent_vec)
                    states.append(state)
                    if i<len(sent_vec):   selected_[i] = 1.0
                    else:   selected_[i] = 0.0
                    if selected_[i] == 1.0:
                        selected_sentIDs.append(sent_IDs[i])
                reward = cnn.get_reward(sent_IDs+ne_sent_IDs,label)
                total_avg_loss+= train_policy.policy_update(states,selected_,reward-avg_reward)
                all_sent_IDs = sent_IDs+ne_sent_IDs
                selected_ = [0.0 for x in range(len(sent_vec+ne_sent_vec))]
                tmp_selected_sent = []
                for i, s in enumerate(all_sent_vec):
                    avg_sent_vec = get_avgVec(selected_, all_sent_vec)
                    state = concat(s, avg_sent_vec)
                    selected_[i] = target_policy.decide_action([state])
                    if selected_[i] == 1.0:
                        tmp_selected_sent.append(all_sent_IDs[i])
                total_selected_reward += cnn.get_reward(tmp_selected_sent, label)
                total_selected_sentences += len(tmp_selected_sent)

        sess.run(copy_network(dest_scope_name="policy_target", src_scope_name="policy_main"))
        total_selected_reward /= len(keys)
        total_avg_loss /= len(keys)
        if total_selected_reward > best_reward:
            best_reward = total_selected_reward
            sess.run(copy_network(dest_scope_name="policy_best", src_scope_name="policy_main"))
        print("Epoch: {}, Reward: {}, Chosen sentences: {}, loss:{}, Time: {}".format(epoch, total_selected_reward,
                                                                                      total_selected_sentences,
                                                                                      total_avg_loss, datetime.now()))
def preRL(f,sess,cnn,train_policy,target_policy, num_epoch,avg_reward):
    print("-- pre-training Instance Selector with fixed CNN model")
    print("Start time: {}".format(datetime.now()))
    sampling_size=3
    keys = list(f.bag_sentIDs.keys())
    print(f.train_data_size,len(keys))
    best_reward = -99999.9
    for epoch in range(num_epoch):
        random.shuffle(keys)  # shuffling bags
        total_selected_sentences = 0
        total_selected_reward=0
        total_batch = int(len(keys) / train_policy.batch_size)+1
        total_avg_loss=0
        for t in range(total_batch):
            s_idx = t*train_policy.batch_size
            e_idx = min((t+1)*train_policy.batch_size,len(keys))
            batch_keys = keys[s_idx:e_idx]
            min_id = f.train_data_size+1
            max_id = 0
            for key in batch_keys:
                for id in f.bag_sentIDs[key]:
                    if id<min_id:   min_id = id
                    if id>max_id:   max_id = id
            cnn.load_data(min_id,max_id)
            for idx,key in enumerate(batch_keys):  # select a bag
                # if idx%100==0:  print(idx)
                # vec_E1 = f.bag_e1[key]
                # vec_E2 = f.bag_e2[key]
                label = f.bag_label[key]
                sent_IDs = f.bag_sentIDs[key]
                sent_vec = cnn.get_sentence_vector(min_id,sent_IDs)
                sampled_reward=[]
                sampled_states=[]
                sampled_selected_=[]
                for _ in range(sampling_size):
                    selected_ = [0.0 for x in range(len(sent_vec))]
                    selected_sentIDs=[]
                    states=[]
                    for i, s in enumerate(sent_vec):
                        avg_sent_vec = get_avgVec(selected_,sent_vec)
                        state = concat(s,avg_sent_vec)
                        states.append(state)
                        selected_[i] = target_policy.get_action([state])
                        if selected_[i] == 1.0:
                            selected_sentIDs.append(sent_IDs[i])
                    if selected_sentIDs:  # Calculate reward from CNN with chosen sentences in the bag
                        sampled_reward.append(cnn.get_reward(selected_sentIDs,label))
                    else:   # If there aren't chosen sentences, then take the average reward
                        sampled_reward.append(avg_reward)
                    sampled_states.append(states)
                    sampled_selected_.append(selected_)
                if sampling_size==1:    tmp_avg_reward=0
                else:   tmp_avg_reward = sum(sampled_reward)/sampling_size
                tmp_loss=0
                for i in range(sampling_size):
                    # update(sess,sampled_states[i],sampled_selected_[i],sampled_reward[i]-tmp_avg_reward,target_policy,0.01)
                    tmp_loss+= train_policy.policy_update(sampled_states[i],sampled_selected_[i],(sampled_reward[i]-tmp_avg_reward))
                total_avg_loss +=(tmp_loss/sampling_size)
                selected_ = [0.0 for x in range(len(sent_vec))]
                tmp_selected_sent=[]
                for i, s in enumerate(sent_vec):
                    avg_sent_vec = get_avgVec(selected_, sent_vec)
                    state = concat(s, avg_sent_vec)
                    selected_[i] = target_policy.decide_action([state])
                    if selected_[i] == 1.0:
                        tmp_selected_sent.append(sent_IDs[i])
                total_selected_reward+=cnn.get_reward(tmp_selected_sent,label)
                total_selected_sentences+=len(tmp_selected_sent)

        sess.run(copy_network(dest_scope_name="policy_target", src_scope_name="policy_main"))
        total_avg_loss/=len(keys)
        total_selected_reward/=len(keys)
        if total_selected_reward>best_reward:
            best_reward = total_selected_reward
            sess.run(copy_network(dest_scope_name="policy_best",src_scope_name="policy_main"))
        print("Epoch: {}, Reward: {:.5f}, Chosen sentences: {}, loss:{:.5f}, Time: {}".format(epoch,total_selected_reward,total_selected_sentences,
                                                                                      total_avg_loss,datetime.now()))
        if epoch%10==0:save_model(epoch,sess,"policy_best")
def RL(sess, f, train_cnn, target_cnn, train_policy, target_policy, num_episode,avg_reward):
    print("-- start Reinforcement Learning")
    print("Start time: {}".format(datetime.now()))
    keys = list(f.bag_sentIDs.keys())
    result_sentIds=[]
    sampling_size=3
    for episode in range(num_episode):
        all_selected_sentIDs=[]
        total_selected_reward = 0
        total_avg_loss=0
        total_batch = int(len(keys) / train_policy.batch_size)+1
        for t in range(total_batch):
            s_idx = t*train_policy.batch_size
            e_idx = min((t+1)*train_policy.batch_size,len(keys))
            batch_keys = keys[s_idx:e_idx]
            min_id=f.train_data_size+1
            max_id=0
            for key in batch_keys:
                for id in f.bag_sentIDs[key]:
                    if id<min_id:   min_id = id
                    if id>max_id:   max_id = id
            target_cnn.load_data(min_id,max_id)
            random.shuffle(batch_keys)
            for idx, key in enumerate(batch_keys):  # Select a bag
                # vec_E1 = f.bag_e1[key]
                # vec_E2 = f.bag_e2[key]
                # if idx%100==0: print(idx)
                sent_IDs = f.bag_sentIDs[key]
                label = f.bag_label[key]
                sent_vec = target_cnn.get_sentence_vector(min_id,sent_IDs)
                sampled_reward = []
                sampled_states = []
                sampled_selected_ = []
                for _ in range(sampling_size):
                    selected_ = [0.0 for x in range(len(sent_vec))]
                    selected_sentIDs = []
                    states = []
                    for i, s in enumerate(sent_vec):
                        avg_sent_vec = get_avgVec(selected_, sent_vec)
                        state = concat(s, avg_sent_vec)
                        # selected_[i] = float(np.random.randint(2))
                        selected_[i] = target_policy.get_action([state])
                        states.append(state)
                        if selected_[i] == 1.0:
                            selected_sentIDs.append(sent_IDs[i])
                    if selected_sentIDs:  # Calculate reward from CNN with chosen sentences in the bag
                        sampled_reward.append(target_cnn.get_reward(selected_sentIDs, label))
                    else:  # If there aren't chosen sentences, then take the average reward
                        sampled_reward.append(avg_reward)
                    sampled_states.append(states)
                    sampled_selected_.append(selected_)
                if sampling_size==1: tmp_avg_reward = 0
                else:   tmp_avg_reward = sum(sampled_reward) / sampling_size
                tmp_loss = 0
                for i in range(sampling_size):
                    tmp_loss+=train_policy.policy_update(sampled_states[i], sampled_selected_[i],
                                                    sampled_reward[i] - tmp_avg_reward)
                total_avg_loss+=(tmp_loss/sampling_size)
                selected_ = [0.0 for x in range(len(sent_vec))]
                states = []
                selected_sentIDs=[]
                for i, s in enumerate(sent_vec):
                    avg_sent_vec = get_avgVec(selected_, sent_vec)
                    state = concat(s, avg_sent_vec)
                    selected_[i] = target_policy.egreedy_action([state],episode)
                    states.append(state)
                    if selected_[i] == 1.0:
                        all_selected_sentIDs.append(sent_IDs[i])
                        selected_sentIDs.append(sent_IDs[i])
                if len(selected_sentIDs)!=0:
                    total_selected_reward+=target_cnn.get_reward(selected_sentIDs,label)
        if len(all_selected_sentIDs)!=0:
            total_selected_reward/=len(all_selected_sentIDs)
        print("Episode: {}, S_reward: {:.5f},T_reward:{:.5f}, selected_sentences: {}, Loss: {:.5f}, time: {}".format(episode,total_selected_reward,avg_reward,
                                                                               len(all_selected_sentIDs),total_avg_loss,datetime.now()))

        train_cnn.update(all_selected_sentIDs,f)
        if episode==num_episode-1:
            result_sentIds=all_selected_sentIDs
        del(all_selected_sentIDs)
        # assign_variables(sess,"policy_target","policy_best",0.01)
        # assign_variables(sess,"cnn_target","cnn_main",0.01)
        # sess.run(copy_network(dest_scope_name="policy_best",src_scope_name="policy_target"))
        sess.run(copy_network(dest_scope_name="cnn_target",src_scope_name="cnn_main"))
        sess.run(copy_network(dest_scope_name="policy_target",src_scope_name="policy_best"))
        target_cnn.save_file(f)
        avg_reward = target_cnn.avg_tot_reward(f.all_labels)
        # print(avg_reward)
        save_model(episode,sess,"cnn_target")
        save_model(episode,sess,"policy_target")
    print(avg_reward)
    return result_sentIds

def save_model(epoch,sess,model):
        vars_RL = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=model)
        saver = tf.train.Saver(vars_RL)
        saver.save(sess,"../model/training/{}_{}".format(model,epoch))

def main():
    f = make_bag_large.readDS()
    f.create_bag()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tf.logging.set_verbosity(tf.logging.WARN)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        c = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_main")
        start_time = time.time()
        avg_reward = c.avg_tot_reward(f.all_labels)
        print("total average reward: {}".format(avg_reward))
        train_policy = InstanceSelector(f,sess,230,0,0.01,net_name="policy_main")
        target_policy = InstanceSelector(f,sess,230,0,0.01,net_name="policy_target")
        best_IS = InstanceSelector(f,sess,230,0,0.01,net_name="policy_best")
        sess.run(tf.global_variables_initializer())
        sess.run(copy_network(dest_scope_name="policy_target",src_scope_name="policy_main"))
        preRL(f,sess,c,train_policy,target_policy,50,avg_reward)
        vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="policy_best")
        saver = tf.train.Saver(vars_policy)
        saver.save(sess,"../model/policy_network/")
        print(time.time()-start_time)

if __name__=="__main__":
    main()
