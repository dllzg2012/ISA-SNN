# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Lambda, Bidirectional, Layer, initializers, regularizers, constraints,Add,Dense,GRU,Conv1D
import tensorflow as tf
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.metrics import mean_squared_error
from nltk.tokenize import word_tokenize
import os
import codecs
from Attention import ISA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class Cross_Attention_SiameseNetwork:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/clinical/train.txt')
        self.test_path = os.path.join(cur, 'data/clinical/process_test300.txt')
        self.bio_test_path = os.path.join(cur, 'data/biosses/embed_test.txt')
        self.cdd_train_path = os.path.join(cur, 'data/CDD/CDD_ful/all_train.txt')
        self.cdd_test_path = os.path.join(cur, 'data/CDD/CDD_ful/process_test.txt')
        self.sentence_file_path = os.path.join(cur, 'data/CDD/CDD_ref/sentence.txt')
        self.root_model_path = os.path.join(cur, 'english_text_model_Siamese')
        tf.gfile.MakeDirs(self.root_model_path)
        self.model_path = os.path.join(self.root_model_path, 'tokenvec_bilstm2_siamese_model.h5')
        self.yaml_path = os.path.join(self.root_model_path, 'tokenvec_bilstm2_siamese_model.yaml')
        self.embed_path=os.path.join(cur, 'pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin')
        self.embed_300_path = os.path.join(cur, 'GoogleNews-vectors-negative300.bin')
        self.TIME_STAMPS = 200#cdd 100;DBMI 200
        self.EMBEDDING_DIM = 400
        self.EPOCHS = 30
        self.BATCH_SIZE = 50
        print("loading the w2v...")
        pretrained_word_model = KeyedVectors.load_word2vec_format(self.embed_path,binary=True)
        self.train_left_datas, self.train_right_datas, self.train_score_datas,self.source_score_list= self.load_data(
            self.train_path, pretrained_word_model,'clinical','train')
        self.test_left_datas, self.test_right_datas, self.test_score_datas,self.source_score_list= self.load_data(
            self.test_path, pretrained_word_model,'clinical','test')




    '''加载数据'''

    def write_txt(self,input_list, file_name):
        fdata = open(file_name, "a")
        fdata.write("\n".join(input_list) + '\n')
    def load_data(self, input_file_path,embed,data_type,task_type):
        input_data = self.read_txt(input_file_path)
        s1_list=[]
        s2_list = []
        score_list=[]
        sorce_score_list=[]
        sentence_list=[]
        def sentence_embed(s1,s2):
            for i, ss in enumerate([s1, s2]):
                words = word_tokenize(ss)
                index = []
                for word in words:
                    if word in embed.vocab:
                        index.append(embed[word])
                    else:
                        index.append(np.zeros(self.EMBEDDING_DIM, dtype=float))
                if i == 0:
                    s1_list.append(np.array(index, dtype=float))
                else:
                    s2_list.append(np.array(index, dtype=float))
            return s1_list,s2_list
        for input in input_data[1:-1]:
            input_split = input.strip().split("\t")
            if data_type=='clinical':
                if len(input_split) == 3:
                    s1 = input_split[0]
                    s2 = input_split[1]
                    label = input_split[2]
                    sentence_list.append(s1+'\t'+s2+'\t'+label)
                    sorce_score_list.append(float(label))
                    score = float(label) / 5
                    score_list.append(score)
                    s1_list, s2_list = sentence_embed(s1, s2)
            elif data_type=='cdd':
                if len(input_split) == 3:
                    s1 = input_split[0]
                    s2 = input_split[1]
                    label = input_split[2]
                    sentence_list.append(s1 + '\t' + s2 + '\t' + label)
                    sorce_score_list.append(float(label))
                    score = (float(label)-1)/4
                    score_list.append(score)
                    s1_list, s2_list = sentence_embed(s1, s2)
        if task_type=='train':
            n_samples=len(s1_list)
            sidx=np.random.permutation(n_samples)
            s1_list = [s1_list[s] for s in sidx]
            s2_list = [s2_list[s] for s in sidx]
            score_list = [score_list[s] for s in sidx]
            sorce_score_list = [sorce_score_list[s] for s in sidx]
        else:
            self.write_txt(sentence_list,self.sentence_file_path)

        train_set=[s1_list,s2_list,score_list,sorce_score_list]

        new_train_set_x1 = np.zeros([len(train_set[0]), self.TIME_STAMPS, self.EMBEDDING_DIM], dtype=float)
        new_train_set_x2 = np.zeros([len(train_set[0]), self.TIME_STAMPS, self.EMBEDDING_DIM], dtype=float)
        new_train_set_y = np.zeros(len(train_set[0]), dtype=float)
        new_source_train_set_y = np.zeros(len(train_set[0]), dtype=float)

        def padding_sentence_vector(x1, x2, y,source_y, new_x1, new_x2, new_y,new_source_y):

            for i, (x1, x2, y,source_y) in enumerate(zip(x1, x2, y,source_y)):
                # whether to remove sentences with length larger than maxlen
                if len(x1) <= self.TIME_STAMPS:
                    new_x1[i, 0:len(x1)] = x1
                    new_y[i] = y
                    new_source_y[i] = source_y
                else:
                    new_x1[i, :, :] = (x1[0:self.TIME_STAMPS:self.EMBEDDING_DIM])
                    new_y[i] = y
                    new_source_y[i] = source_y
                if len(x2) <= self.TIME_STAMPS:
                    new_x2[i, 0:len(x2)] = x2
                    new_y[i] = y
                    new_source_y[i] = source_y
                else:
                    new_x2[i, :, :] = (x2[0:self.TIME_STAMPS:self.EMBEDDING_DIM])
                    new_y[i] = y
                    new_source_y[i] = source_y
            new_set = [new_x1, new_x2, new_y,new_source_y]
            del new_x1, new_x2, new_y,new_source_y
            return new_set

        train_set = padding_sentence_vector(train_set[0], train_set[1], train_set[2],train_set[3], new_train_set_x1,new_train_set_x2,new_train_set_y,new_source_train_set_y)
        return train_set

    def read_txt(self, filename):
        fileData = codecs.open(filename, "r", encoding='ascii', errors='ignore')
        readfile = fileData.readlines()
        return readfile

    '''将数据转换成keras所需的格式'''

    def modify_train_data(self):
        y_train = self.train_score_datas
        left_x_train = self.train_left_datas
        right_x_train = self.train_right_datas

        y_train = np.expand_dims(y_train, 2)
        return left_x_train, right_x_train, y_train

    def modify_test_data(self):
        y_test = self.source_score_list
        left_x_test = self.test_left_datas
        right_x_test = self.test_right_datas

        y_test = np.expand_dims(y_test, 2)
        return left_x_test, right_x_test, y_test

    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''

    def exponent_neg_manhattan_distance(self, sent_left, sent_right):
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))


    '''搭建编码层网络,用于权重共享'''

    def create_base_network(self, input_shape,use_seq):
        input = Input(shape=input_shape)

        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(32, return_sequences=use_seq))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''

    def bilstm_siamese_model(self):

        left_input = Input(shape=(self.TIME_STAMPS, self.EMBEDDING_DIM), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS, self.EMBEDDING_DIM), dtype='float32')

        '''interactive_self_attention process'''
        def interactive_self_attention():
            shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM), use_seq=True)
            left_output = shared_lstm(left_input)
            right_output = shared_lstm(right_input)
            att_left=ISA(self.TIME_STAMPS)([left_output, right_output])
            att_right = ISA(self.TIME_STAMPS)([right_output,left_output])


            return att_left, att_right


        left_output,right_output=interactive_self_attention()

        distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        model = Model([left_input, right_input], [distance])
        model.compile(loss='mean_squared_error',
                      optimizer='nadam',
                      metrics=['acc'])  # cosine_proximity
        model.summary()
        return model

    '''预测结果'''

    def predict_result(self, model, left_data, right_data):
        y_pred = model.predict([left_data, right_data])
        return y_pred

    '''计算误差'''

    def compute_average_error(self, pred_list, label_list):
        error_list = []
        for i in range(len(pred_list)):
            error_list.append(abs(float(label_list[i]) - float(pred_list[i])))
        average_error = np.mean(error_list)
        max_error = max(error_list)
        min_error = min(error_list)
        return average_error, max_error, min_error

    '''画误差图，预测值，标签值'''

    def draw_error_graph(self, pred_list, label_list):
        def make_list(input_list):
            list, index = [], []
            for i, input in enumerate(input_list):
                list.append(float(input))
                index.append(i)
            return list, index

        pr_list, pr_index = make_list(pred_list)
        la_list, la_index = make_list(label_list)
        plt.plot(pr_index, pr_list, linestyle='-.', label="PRED-LIST", color='blue')
        plt.plot(la_index, la_list, linestyle='--', label="LABEL-LIST", color='red')
        plt.legend()
        plt.show()

    '''评价结果'''

    def compute_predict_result(self, pred_list, label_list):
        MSE = mean_squared_error(label_list, pred_list)
        spearmanr = stats.spearmanr(pred_list, label_list)
        pearsonr = stats.pearsonr(pred_list, label_list)
        average_error, max_error, min_error = self.compute_average_error(pred_list, label_list)
        return pearsonr, average_error, max_error, min_error, spearmanr, MSE

    '''训练模型'''

    def train_model(self):
        left_x_train, right_x_train, y_train = self.modify_train_data()
        model = self.bilstm_siamese_model()
        history = model.fit(
            x=[left_x_train, right_x_train],
            y=y_train,
            validation_split=0.1,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
        )
        #self.save_model(model)
        return model

    '''保存模型'''
    def save_model(self,model):
        model.save_weights(self.model_path)
        yaml_string = model.to_yaml()
        open(self.yaml_path, 'w').write(yaml_string)
    '''加载模型'''
    def load_model(self):
        model = model_from_yaml(open(self.yaml_path).read(),
                                custom_objects={'ISA': ISA(step_dim=self.TIME_STAMPS)})
        model.load_weights(self.model_path, by_name=True)
        return model

    '''测试模型'''

    def test_model(self, model):
        print('Testing the model ...')
        left_x_test, right_x_test, y_test = self.modify_test_data()
        pred_list = self.predict_result(model, left_x_test, right_x_test)
        pred_label_list = []
        label_list = []
        for pred in pred_list:
            pred = str(pred)
            pred = pred.lstrip("[")
            pred = pred.rstrip("]")
            pred = float(pred)
            # pred=round((pred),1)
            pred = round((pred * 5.0), 1)
            #pred = round((pred * 4.0+1), 1)
            pred_label_list.append(pred)
        for label in y_test:
            label_list.append(round((float(label)), 1))
            # label_list.append(round(float(label)*5,1))
            # label_list.append(round((float(label)*4.0+1),1))
        print(pred_label_list)
        print(label_list)
        pearsonr, average_error, max_error, min_error, spearmanr, MSE = self.compute_predict_result(pred_label_list,
                                                                                                    label_list)
        print("Pearson correlation coefficient:", pearsonr)
        print("Spearman correlation coefficient:", spearmanr)
        print("MSE:", MSE)
        print("Average error rate:", average_error)
        print("Max error rate:", max_error)
        print("Min error rate:", min_error)
        self.draw_error_graph(pred_label_list, label_list)


handler = Cross_Attention_SiameseNetwork()
model = handler.train_model()
#model=handler.load_model()
handler.test_model(model)
