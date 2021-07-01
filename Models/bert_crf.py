# *- coding: utf-8 -*-

# =================================
# time: 2021.6.04
# author: @唐志林
# function: 模型适配
# =================================

import re
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from datetime import datetime
from sklearn.metrics import f1_score
from transformers import BertTokenizer, TFBertModel
from DatasetProcess import data_process

epochs = 4
max_len = 128
batch_size = 16
hidden_dim = 200

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

vocab_size = len(tokenizer.get_vocab())

time_month = datetime.now().month
time_day = datetime.now().day

log_dir = f'../ModelCkpt/model_save/logs/'
# bert_crf_ckpt = f'../ModelCkpt/bert_crf_save2pb/pb_model_{time_month}_{time_day}/'
bert_ckpt = f'../ModelCkpt/bert_save2pb/pb_model_{time_month}_{time_day}/'
bert_crf_ckpt = '../ModelCkpt/bert_crf_save2pb/tfs/1/'

dirs = [log_dir, bert_crf_ckpt, bert_ckpt]
for file in dirs:
    if not os.path.exists(file):
        os.makedirs(file)


# todo 适配对话数据的编码
num_class = len(['O', 'B-IND', 'B-QT', 'B-PST', 'B-PS']) + 1


def labels4seq(data, id2seq=False):
    # 适配羽白任务导向对话数据的编码
    if not id2seq:
        label_seq = {
            'O': 1, 'B-IND': 2, 'B-QT': 3, 'B-PST': 4, 'B-PS': 5}
    else:
        label_seq = {
            '1': 'O', '2': 'B-IND', '3': 'B-QT', '4': 'B-PST', '5': 'B-PS', '0': '0'}

    label = [label_seq[i] for i in data]

    return label


def dataset_generator(data):
    ids, masks, tokens, labels, labels_length = [], [], [], [], []

    data = data.sample(frac=1.0)

    for num, (_, d) in enumerate(data.iterrows()):
        content = ''.join(d['content'])

        inputs = tokenizer.encode_plus(content,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       padding='max_length',
                                       return_token_type_ids=True)

        input_id = inputs['input_ids']
        input_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        label_length = len(d['label'])
        # 加入label_length的目的仅限于后续将content, label和predict值切分出来，在训练过程中不涉及

        label = labels4seq(d['label'])
        label = label + (max_len - len(label)) * [0]

        ids.append(input_id)
        masks.append(input_mask)
        tokens.append(token_type_ids)
        labels.append(label)
        labels_length.append([label_length])

        if len(ids) == batch_size or _ == len(data):
            yield {
                'ids': tf.constant(ids, dtype=tf.int32),
                'masks': tf.constant(masks, dtype=tf.int32),
                'tokens': tf.constant(tokens, dtype=tf.int32),
                'labels': tf.constant(labels, dtype=tf.int32),
                'labels_length': tf.constant(labels_length, dtype=tf.int32)
            }
            ids, masks, tokens, labels, labels_length = [], [], [], [], []


class MyBertCrf(tf.keras.Model):
    def __init__(self, use_crf, input_dim, output_dim):
        super(MyBertCrf, self).__init__(use_crf, input_dim, output_dim)
        self.use_crf = use_crf
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bert = TFBertModel.from_pretrained('hfl/chinese-bert-wwm-ext')

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(self.output_dim)
        self.other_params = tf.Variable(tf.random.uniform(shape=(output_dim, output_dim)))

    @tf.function(input_signature=[(tf.TensorSpec([None, 128], name='ids', dtype=tf.int32),
                                  tf.TensorSpec([None, 128], name='mask', dtype=tf.int32),
                                  tf.TensorSpec([None, 128], name='tokens', dtype=tf.int32),
                                  tf.TensorSpec([None, 128], name='target', dtype=tf.int32))])
    def call(self, batch_data):
        ids, masks, tokens, target = batch_data
        input_seq_len = tf.cast(tf.reduce_sum(masks, axis=1), dtype=tf.int32)

        hidden = self.bert(ids, masks, tokens)[0]
        dropout_inputs = self.dropout(hidden, 1)
        logistic_seq = self.dense(dropout_inputs)
        if self.use_crf:
            log_likelihood, self.other_params = tfa.text.crf.crf_log_likelihood(logistic_seq,
                                                                                target,
                                                                                input_seq_len,
                                                                                self.other_params)
            decode_predict, crf_scores = tfa.text.crf_decode(logistic_seq, self.other_params , input_seq_len)

            return decode_predict, log_likelihood, crf_scores
        else:
            prob_seq = tf.nn.softmax(logistic_seq)

            return prob_seq, None, None


def get_loss(log_likelihood):
    # have crf
    loss = -tf.reduce_mean(log_likelihood)

    return loss


def get_loss_2(t, p):
    # no crf
    loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true=t, y_pred=p)
    loss_value = tf.reduce_mean(loss_value)

    return loss_value


def get_f1_score(labels, predicts, use_crf):
    if not use_crf:
        predicts = tf.argmax(predicts, axis=-1)

    labels = np.array(labels)
    predicts = np.array(predicts)
    l_p = zip(labels, predicts)

    f1_value = np.array([f1_score(y_true=l, y_pred=p, average="macro") for l, p in l_p]).mean()

    return f1_value


def fit_dataset(dataset, use_crf, input_dim, output_dim, fit=True):
    # bert层的学习率与其他层的学习率要区分开来
    dataset = dataset_generator(dataset)

    bert_crf = MyBertCrf(use_crf, input_dim, output_dim)

    opti_bert = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.95)
    opti_other = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.95)

    if use_crf:
        model_ckpt = bert_crf_ckpt
    else:
        model_ckpt = bert_ckpt

    def fit_models(batch_data):
        params_bert = []
        params_other = []

        ids = batch_data['ids']
        masks = batch_data['masks']
        tokens = batch_data['tokens']
        target = batch_data['labels']

        with tf.GradientTape() as tp:
            predict_seq, log_likelihood, crf_scores = bert_crf((ids, masks, tokens, target))
            if use_crf:
                loss_value = get_loss(log_likelihood)
            else:
                loss_value = get_loss_2(target, predict_seq)

            for var in bert_crf.trainable_variables:
                model_name = var.name
                none_bert_layer =  ['tf_bert_model/bert/pooler/dense/kernel:0',
                                    'tf_bert_model/bert/pooler/dense/bias:0']

                if model_name in none_bert_layer:
                    pass
                elif model_name.startswith('tf_bert_model'):
                    params_bert.append(var)
                else:
                    params_other.append(var)

        params_all = tp.gradient(loss_value, [params_bert, params_other])
        gradients_bert = params_all[0]
        gradients_other = params_all[1]

        opti_other.apply_gradients(zip(gradients_other, params_other))

        opti_bert.apply_gradients(zip(gradients_bert, params_bert))

        return loss_value, predict_seq, target

    if fit:
        # fit
        for _, data in enumerate(dataset):
            loss, predicts, labels = fit_models(batch_data=data)

            if _ % 5 == 0:
                print(f"step: {_}, loss_value: {loss}")

            if _ % 20 == 0:
                f1_value = get_f1_score(labels=labels, predicts=predicts, use_crf=use_crf)
                with open(log_dir + 'fit_logs.txt', 'a') as f:
                    # 'a'  要求写入字符
                    # 'wb' 要求写入字节(str.encode(str))
                    log = f"date: {time_month}-{time_day}, step: {_}, loss{loss}, f1_score: {f1_value} \n"
                    f.write(log)

        bert_crf.save(filepath=model_ckpt)

    else:
        # valid
        valid_pre_label = pd.DataFrame()
        bert_crf = tf.saved_model.load(model_ckpt)

        for num, inputs in enumerate(dataset):
            valid_id = inputs['ids']
            valid_mask = inputs['masks']
            valid_token = inputs['tokens']
            valid_target = inputs['labels']
            valid_pred, _, _ = bert_crf((valid_id, valid_mask, valid_token, valid_target))

            f1_value = get_f1_score(labels=valid_target, predicts=valid_pred, use_crf=use_crf)

            with open(log_dir + 'valid_logs.txt', 'a') as f:
                # 'a'  要求写入字符
                # 'wb' 要求写入字节(str.encode(str))
                log = f"date: {time_month}-{time_day}, batch: {num}, f1_score: {f1_value} \n"
                f.write(log)

        return valid_pre_label


def get_predict_data(content):
    """将输入的问句初步token处理
    """
    inputs = tokenizer.encode_plus(content,
                                   add_special_tokens=True,
                                   max_length=max_len,
                                   padding='max_length',
                                   return_token_type_ids=True)

    input_id = tf.constant([inputs['input_ids']], dtype=tf.int32)
    input_mask = tf.constant([inputs['attention_mask']], dtype=tf.int32)
    token_type_ids = tf.constant([inputs["token_type_ids"]], dtype=tf.int32)

    label_length = len(content)
    label = tf.constant([max_len * [0]], dtype=tf.int32)
    input_data = (input_id, input_mask, token_type_ids, label)

    return input_data, label_length


def predict_data(input_id, input_mask, token_type_ids, label, crf=True):
    """预测处理后的输入问句
    """
    if crf:
        model_ckpt = bert_crf_ckpt
    else:
        model_ckpt = bert_ckpt

    bert_crf = tf.saved_model.load(model_ckpt)
    predict_label, _, _ = bert_crf.call((input_id, input_mask, token_type_ids, label))

    if not crf:
        predict_label = tf.argmax(predict_label, axis=-1)

    return predict_label


def get_keywords_label(predict_label, label_length, input_id):
    """获取模型预测的结果
    keywords: 指序列标注保留的关键数据
    label: 用于配合keywords在get_slot()中将关键数据填入到对应的插槽中
    """
    predict_label = np.array(predict_label)[0][:label_length]
    predict_label_mask = [0 if p <= 1 else 1 for p in predict_label]
    # 根据keywords的编码获取input_id内部的关键词id
    input_id_mask = np.array(input_id)[0][1: label_length + 1]

    keywords_id = predict_label_mask * input_id_mask
    keywords = tokenizer.decode(keywords_id)

    # 将会根据关键词内部的id和预测值的关键词编码获取到对应槽的值
    return keywords, predict_label


def get_slot(keywords, predict_label):
    """考虑到数据特点关键词是以词组的形式出现, 为了准确的找到关键词对应的代码(a,b,c,d)
    因此使用zip数组将keywords和predict连接在一起
    设置一个字典, 代码(a,b,c,d)作为key, 值为一个列表, 因此一个代码可以添加多个对应关键词
    """
    keywords = ''.join(i if i != ' ' else '' for i in keywords)
    keywords_list = re.findall(pattern='[\u4e00-\u9fa5(（）)]+', string=keywords)

    predict_label = ''.join([str(i) if i !=' ' else '' for i in list(predict_label)])
    predict_list = re.findall(pattern='[2-9]+', string=predict_label)

    seq_key = {
        '2': 'industry', '3': 'question_type', '4': 'process_type', '5': 'process'}
    key_words = {
        'industry': [], 'question_type': [], 'process_type': [], 'process': []}

    for key, words in zip(predict_list, keywords_list):
        key_words[seq_key[str(key[0])]].append(words)

    return key_words


def predict(content, crf=True):
    """1: 将输入的问句进行初步token处理---get_predict_data(）
    2: 模型预测获取序列标注结果---predict_data()
    3: 将序列标注结果进行处理获取关键词和关键词对应槽值---get_keywords_label()
    4: 将关键词插入到对应的槽里面---get_slot()
    """
    input_data, label_length =  get_predict_data(content)
    input_id, input_mask, token_type_ids, label = input_data
    predict_label = predict_data(input_id, input_mask, token_type_ids, label, crf=crf)

    keywords, predict_label = get_keywords_label(predict_label=predict_label,
                                                 label_length=label_length,
                                                 input_id=input_id)

    keywords = get_slot(keywords, predict_label)

    return keywords


def fit():
    # todo 待改进
    test_data = data_process.test_flow_dataset()
    train = test_data[:11200]
    test = test_data[11200:]
    fit_dataset(dataset=train, use_crf=True, input_dim=vocab_size, output_dim=num_class, fit=True)


def valid():
    # todo 待改进
    test_data = data_process.test_flow_dataset()
    train = test_data[:11200]
    test = test_data[11200:]
    fit_dataset(dataset=test, use_crf=True, input_dim=vocab_size, output_dim=num_class, fit=False)

print(predict(content='你知道铸造行业树脂砂铸造类型中的焊接对于环境有那些危害'))







