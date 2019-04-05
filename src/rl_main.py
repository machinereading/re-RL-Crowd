import policy_network
import pycnn
import make_bag_large
import time
import tensorflow as tf
import os
if __name__=="__main__":
    f = make_bag_large.readDS()
    f.create_bag()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tf.logging.set_verbosity(tf.logging.WARN)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        start_time = time.time()
        print("-- build & initialize models")
        main_cnn = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_main")
        target_cnn = pycnn.CNN(sess,f.num_classes,f.sequence_length,0.01,"cnn_target")
        main_policy = policy_network.InstanceSelector(f, sess,230,0, 0.01, "policy_best")
        target_policy = policy_network.InstanceSelector(f, sess, 230,0,0.01, "policy_target")
        main_cnn.build_model()
        target_cnn.build_model()
        sess.run(tf.global_variables_initializer())

        print("-- load main cnn & policy models")
        vars_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="cnn_main")
        saver = tf.train.Saver(vars_cnn)
        saver.restore(sess,"../model/cnn/cnn_main_30000_100")
        sess.run(policy_network.copy_network(dest_scope_name="cnn_target",src_scope_name="cnn_main"))
        labels = f.all_labels
        avg_reward = target_cnn.avg_tot_reward(labels)
        print("total average reward: {}".format(avg_reward))
        vars_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_best")
        saver = tf.train.Saver(vars_policy)
        saver.restore(sess, "../model/policy_network/policy_main")
        sess.run(policy_network.copy_network(dest_scope_name="policy_target",src_scope_name="policy_best"))

        results= policy_network.RL(sess,f,main_cnn,target_cnn,main_policy,target_policy,25,avg_reward)
        print("-- complete training RL")

        vars_RL = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="policy_target")
        saver = tf.train.Saver(vars_RL)
        saver.save(sess,"../model/policy_network/policy_target")
        vars_cnn = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="cnn_target")
        saver = tf.train.Saver(vars_cnn)
        saver.save(sess,"../model/cnn/cnn_target")

        fw = open("../result/selected_sentences.txt",'w',encoding='utf-8')
        for i in results:
            fw.write(str(i)+"\n")

        print("[DONE]")

