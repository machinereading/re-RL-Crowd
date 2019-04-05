import csv
import random
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
class readDS():
    def __init__(self,train_data):
        self.train_data = train_data
        self.test_data = "../data/test_data.tsv"
        self.properties = "../data/ds_label_properties44.txt"
        self.entity2id_path = "../data/entity2id.txt"
        self.entity_vec_path = "../data/entity2vec.bern"
        self.sequence_length=80
        self.wv_model = KeyedVectors.load_word2vec_format("../data/w2v_1226.vec")
        self.embedding_size = self.wv_model.vector_size
        # self.embedding_size=100
        print("-- complete loading word embedding")
        self.positionVec = np.zeros((122,5),float)
        self.entity_vec = pd.read_csv(self.entity_vec_path, header=None, delim_whitespace=True).values
        with open("../data/positionVec.data",'r') as f:
            for i, r in enumerate(f.readlines()):
                r = r.strip().split("\t")
                self.positionVec[i] = np.array([float(x) for x in r])
        self.true_tags=[]
        with open(self.properties,'r',encoding='utf-8') as property,open(self.train_data,'r',encoding='utf-8') as trainf,open(self.entity2id_path,'r',encoding='utf-8') as entity:
            self.relation2id = property.read().rstrip().split("\n")
            self.num_classes = len(self.relation2id)
            self.train_data_size = len(trainf.readlines())
            self.entity2id = []
            self.entity_set=set()
            for r in entity.readlines():
                r=r.split("\t")
                self.entity2id.append(r[0])
                self.entity_set.add(r[0])
        print(self.train_data,self.train_data_size)
    def load_true_tags(self):
        true_tags=[]
        with open(self.train_data,'r',encoding='utf-8') as f:
            for line in csv.reader(f,delimiter='\t'):
                if line[-1]=="T":   true_tags.append([1.0])
                else:   true_tags.append([0.0])
        return true_tags
    def id2relation(self,ids):
        labels=[]
        for id in ids:
            labels.append(self.relation2id[id])
        return labels
    def pos_embed(self,x):
        if x< -60:  return 0
        if x>=-60 and x<=60:  return x+61
        if x>60:    return 121

    def create_bag(self):
        self.bag_sentIDs={}
        self.bag_label={}
        self.bag_e1={}
        self.bag_e2={}
        self.all_labels=[]
        for idx, line in enumerate(csv.reader(open(self.train_data,'r',encoding='utf-8'),delimiter='\t')):
            relation = self.relation2id.index(line[1])
            self.all_labels.append(relation)
            key = line[5]+","+line[6]+","+str(relation)
            if key in self.bag_sentIDs:
                self.bag_sentIDs[key].append(idx)
            else:
                self.bag_sentIDs[key] = [idx]
                self.bag_label[key] = relation
                self.bag_e1[key] = line[5].split("/")[0]
                self.bag_e2[key] = line[6].split("/")[0]

    def get_vec_entity(self,entity):
        entity_id = self.entity2id.index(entity)
        return self.entity_vec[entity_id]


    def read_batch_data_with_id(self,sentIDs):
        # print("--reading raw data from {} to {} for {}".format(st,en,flag))
        st = sentIDs[0]
        en = sentIDs[-1]
        with open(self.train_data,'r',encoding='utf-8') as f:
            sentence_data = list(csv.reader(f,delimiter='\t'))[st:en]
        w2v_sentences=[]
        relations = []
        for idx, line in enumerate(sentence_data):
            sent_id = idx+st
            if sent_id in sentIDs:
                sentence = line[0]
                s_tokens = sentence.rstrip().split()
                relations.append(self.relation2id.index(line[1]))
                tmp_s = np.zeros((self.sequence_length, self.embedding_size + 10), dtype=float)
                p1 = s_tokens.index("<<_sbj_>>")
                p2 = s_tokens.index("<<_obj_>>")
                s_tokens[p1] = line[5]
                s_tokens[p2] = line[6]
                for i, word in enumerate(s_tokens):
                    if word not in self.wv_model:   continue
                    word_vec = self.wv_model[word]
                    pE1 = self.positionVec[self.pos_embed(p1 - i)]
                    pE2 = self.positionVec[self.pos_embed(p2 - i)]
                    tmp = np.append(word_vec, pE1)
                    tmp = np.append(tmp, pE2)
                    tmp_s[i] = tmp
                w2v_sentences.append(tmp_s)

        return w2v_sentences, relations

    def read_tag(self,sentIDs):
        tags=[]
        for sent_idx in sentIDs:
            tags.append(self.true_tags[sent_idx])
        return tags


    def read_relations(self,st,en):
        with open(self.train_data,'r',encoding='utf-8') as train_data:
            sentence_data = list(csv.reader(train_data,delimiter='\t'))[st:en]
        relations = []
        for line in sentence_data:
            relations.append(self.relation2id.index(line[1]))
        return relations

    def read_batch_data(self,flag,st,en):
        # print("--reading raw data from {} to {} for {}".format(st,en,flag))
        if flag=="train":
            with open(self.train_data,'r',encoding='utf-8') as train_file:
                sentence_data = list(csv.reader(train_file,delimiter='\t'))[st:en]
        elif flag=="test":
            with open(self.test_data,'r',encoding='utf-8') as test_file:
                sentence_data = list(csv.reader(test_file,delimiter='\t'))[st:en]
        else:   return

        w2v_sentences=[]
        relations=[]
        for idx, line in enumerate(sentence_data):
            sentence = line[0]
            s_tokens = sentence.rstrip().split()
            for i,token in enumerate(s_tokens): s_tokens[i] = token.strip()
            if len(s_tokens)>self.sequence_length:  continue
            try:
                relations.append(self.relation2id.index(line[1]))
            except:
                print(line)
                exit()
            tmp_s = np.zeros((self.sequence_length,self.embedding_size+10),dtype=float)
            p1 = s_tokens.index("<<_sbj_>>")
            p2 = s_tokens.index("<<_obj_>>")
            s_tokens[p1] = line[5]
            s_tokens[p2] = line[6]
            for i, word in enumerate(s_tokens):
                if word not in self.wv_model:   continue
                word_vec = self.wv_model[word]
                pE1 = self.positionVec[self.pos_embed(p1-i)]
                pE2 = self.positionVec[self.pos_embed(p2-i)]
                tmp = np.append(word_vec,pE1)
                tmp = np.append(tmp,pE2)
                tmp_s[i] = tmp
            w2v_sentences.append(tmp_s)

        return w2v_sentences, relations

if __name__ =="__main__":
    f = readDS()
    f.create_bag()
    print(len(f.bag_sentIDs.keys()))
