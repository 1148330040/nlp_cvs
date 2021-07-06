# *- coding: utf-8 -*-

# =================================
# time: 2021.7.02
# author: @唐志林
# function: 相似度功能添加
# =================================

import os
import numpy as np
import pandas as pd

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.models import build_transformer_model


pd.set_option('display.max_columns', None)

pretrained_path = '/mnt/tang_cvs/FinTurningModel/simbert/chinese_simbert_L-6_H-384_A-12/'

bert_config_path = os.path.join(pretrained_path, 'bert_config.json')
bert_model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
bert_vocab_path = os.path.join(pretrained_path, 'vocab.txt')

tokenizer = Tokenizer(token_dict=bert_vocab_path, do_lower_case=True)

bert = build_transformer_model(
    bert_config_path,
    bert_model_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

bert = keras.models.Model(bert.model.inputs, bert.model.outputs[0])


def get_Top1similarity_word(word, similarity_words):
    """word: 错误槽数据
    similarity_words: 待匹配槽数据列表
    """
    word = word
    similarity_words = list(similarity_words)
    for w in similarity_words:
        if type(w) != str:
            similarity_words.remove(w)
    print(f"错误关键词: {word}")
    print(f"对比相似度的候选关键词: {similarity_words}")

    token_ids_a = sequence_padding([tokenizer.encode(v, maxlen=50)[0] for v in similarity_words])

    token_ids_vec_a = bert.predict(
        [token_ids_a, np.zeros_like(token_ids_a)], verbose=True, batch_size=1
    )

    a_vec = token_ids_vec_a / (token_ids_vec_a ** 2).sum(axis=1, keepdims=True) ** 0.5

    vec_all = a_vec.reshape(-1, 384)

    def get_word_index(text):
        token_ids, segment_ids = tokenizer.encode(text)
        vec = bert.predict([[token_ids], [segment_ids]])[0]
        vec /= (vec**2).sum()**0.5
        sims = np.dot(vec_all, vec)

        return np.nanargmax(sims)

    index = get_word_index(word)

    return similarity_words[index]


