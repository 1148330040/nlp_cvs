# *- coding: utf-8 -*-

# =================================
# time: 2021.6.04
# author: @唐志林
# function: 数据整合处理
# =================================
import os
import re
import yaml

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from SqlLink import sql_interactive

pd.set_option('display.max_columns', None)
config_path = '../Config/config.yml'

r"""
目前来说, 对于ds4和ds5它们的问题类型即question_type只有一种'绿色化提升'
因此对于ds4来说只需要考虑两个槽即可, 对于ds5只需要考虑三个槽即可
就当前的情况来说, 有两种数据类型, 四种种槽
因此处理的流程就是:
1: 确定输入语句归属于哪一种数据类型
2: 确定了数据类型就需要找到对应的两个或者三个槽
3: 对于ds4类型的数据只需要找到槽二即可获取对应的回复
   对于ds5类型数据则至少需要找到槽二和曹三即可获取对应的回复(这里不考虑重复的数据)
4: 如果找到的槽可以在原数据中找到，则直接依据key返回对应的answer
5: 否则考虑使用相似度进行确认，抑或直接给出对应槽的选择

ds4 = None # 数据类型一
ds5 = None # 数据类型二

mask1 = None # 槽一: 控制行业
mask2 = None # 槽二: 控制问题类型
mask3 = None # 槽三: 控制工艺类型
mask4 = None # 槽四: 控制工艺

a,b,c,d代替行业，问题类型，工艺类型，工艺
"""

def get_QA_dataset():
    dataset_path = '../Dataset/dataset_dz/'

    nan_files = ['羟丙基甲基纤维素制造', '氧化铝', '炼油与石油化工']
    # nan_problems = ['节能环保政策', '智能制造政策', '智能化提升']
    # 目前数据不存在这三种问题类型

    dataset_4col = pd.DataFrame()
    # 根据需要的列的数目区分
    dataset_5col = pd.DataFrame()

    for file in os.listdir(dataset_path):
        data = pd.read_excel(dataset_path + file, sheet_name='内容')
        mid = True

        for nan_file in nan_files:
            if nan_file in file:
                mid = False
                break
        if mid:
            if len(data.columns) == 5:
                data.columns = ['_', 'industry', 'question_type',
                                'process', 'answer']
                data.reset_index(inplace=True)

                dataset_4col = pd.concat([dataset_4col, data[['industry', 'question_type',
                                                        'process', 'answer']]])
            else:
                data.columns = ['_', 'industry', 'question_type', 'process_type',
                                'process', 'answer']
                data.reset_index(inplace=True)

                dataset_5col = pd.concat([dataset_5col, data[['industry', 'question_type', 'process_type',
                                                        'process', 'answer']]])

    return dataset_4col, dataset_5col


def get_template_dataset():
    # 第一步: 获取模板归属的数据类型
    # 第二步: 抽取对应数据类型的数据
    # 第三步: 替换模板中指定的关键词代码

    data_template = pd.DataFrame(open(file='../Dataset/输入语句模板.txt'), columns=['template'])

    data_template['template'] = data_template['template'].apply(
        lambda x: x.strip()
    )

    # 模板中有c存在, 则必须从ds5中抽取数据
    # 模板中如果没有c存在, 则可以从ds4或者ds5中抽取数据
    data_template['type'] = data_template['template'].apply(
        lambda x: '1' if 'c' in x else '0'
    )

    t_5 = data_template.loc[data_template['type'] == '1']
    t_4and5 = data_template.loc[data_template['type'] == '0']

    return t_5, t_4and5


def fill_template(d5, d45, t5, t45):
    """
    :param d4: Q&A数据ds4
    :param d5: Q&A数据ds5
    :param t5: template数据 针对于 no_c 数据
    :param t45: template数据 仅针对于have_c 数据
    :return: fill template Dataset
    """
    def fill_t5(ds5, template_5):
        # have c
        content = []
        label = []
        for epoch in range(50):
            for _, d in template_5.iterrows():
                t = d['template']
                a_count = str(t).count('a')
                b_count = str(t).count('b')
                c_count = str(t).count('c')
                d_count = str(t).count('d')

                max_count = max([a_count, b_count, c_count, d_count])

                ds5_ = ds5.sample(n=max_count)

                a_values = list(ds5_['industry'].values)
                b_values = list(ds5_['question_type'].values)
                c_values = list(ds5_['process_type'].values)
                d_values = list(ds5_['process'].values)

                fill_data = zip(a_values, b_values, c_values, d_values)

                content_ = str(t)
                label_ = str(t)

                for a_value, b_value, c_value, d_value in fill_data:
                    content_ = str(content_).replace('a', a_value, 1)\
                                            .replace('b', b_value, 1)\
                                            .replace('c', c_value, 1)\
                                            .replace('d', d_value, 1)

                    label_ = str(label_).replace('a', str(len(a_value) * 'A'), 1)\
                                        .replace('b', str(len(b_value) * 'B'), 1)\
                                        .replace('c', str(len(c_value) * 'C'), 1)\
                                        .replace('d', str(len(d_value) * 'D'), 1)

                content.append(content_)
                label.append(list(label_))

        content = pd.DataFrame(np.array(content).reshape(-1, 1), columns=['content'])
        print()
        label = pd.DataFrame(np.array(label).reshape(-1, 1), columns=['label'])
        data_5 = pd.concat([content, label], axis=1)

        return data_5

    def fill_t45(ds45, template_45):
        # no c
        content = []
        label = []
        for epoch in range(50):
            for _, d in template_45.iterrows():
                t = d['template']
                a_count = str(t).count('a')
                b_count = str(t).count('b')
                d_count = str(t).count('d')

                max_count = max([a_count, b_count, d_count])

                ds45_ = ds45.sample(n=max_count)

                a_values = list(ds45_['industry'].values)
                b_values = list(ds45_['question_type'].values)
                d_values = list(ds45_['process'].values)
                fill_data = zip(a_values, b_values, d_values)

                content_ = str(t)
                label_ = str(t)
                for a_value, b_value, d_value in fill_data:
                    content_ = str(content_).replace('a', a_value, 1)\
                                            .replace('b', b_value, 1)\
                                            .replace('d', d_value, 1)

                    label_ = str(label_).replace('a', str(len(a_value) * 'A'), 1)\
                                        .replace('b', str(len(b_value) * 'B'), 1)\
                                        .replace('d', str(len(d_value) * 'D'), 1)

                content.append(content_)
                label.append(list(label_))

        content = pd.DataFrame(np.array(content).reshape(-1, 1), columns=['content'])
        print("")
        label = pd.DataFrame(np.array(label).reshape(-1, 1), columns=['label'])
        data_d45 = pd.concat([content, label], axis=1)

        return data_d45

    content_d5 = fill_t5(ds5=d5, template_5=t5)
    content_d45 = fill_t45(ds45=d45, template_45=t45)
    dataset = pd.concat([content_d5, content_d45])
    dataset.index = np.arange(len(dataset))

    dataset['content_len'] = dataset['content'].apply(
        lambda x: len(x)
    )

    dataset['label_len'] = dataset['label'].apply(
        lambda x: len(x)
    )

    dataset['consistent_len'] = dataset['content_len'] - dataset['label_len']

    dataset['label'] = dataset['label'].apply(
        lambda x: ''.join(x)
    )

    return dataset


def save_cvs_dataset():
    with open(config_path) as f:
        config = yaml.load(f)
        mysql_params = config['mysql_db']

    pass_word = mysql_params['pass_word']
    host = mysql_params['host']
    port = mysql_params['port']
    db_name = mysql_params['data_base']

    d4, d5 = get_QA_dataset()
    t5, t45 = get_template_dataset()

    d4 = d4[['industry', 'question_type', 'process', 'answer']]
    d5 = d5[['industry', 'question_type', 'process_type', 'process', 'answer']]
    d4.dropna(inplace=True)
    d5.dropna(inplace=True)

    d45 = pd.concat([d4, d5[['industry', 'question_type', 'process']]])

    cvs_dataset = fill_template(d5, d45, t5, t45)

    engine = create_engine(f"mysql+pymysql://root:{pass_word}@{host}:{port}/{db_name}?charset=utf8")

    cvs_dataset.to_sql(name='test_data', con=engine, if_exists='replace',
                 index=False, index_label=False)

    d5.to_sql(name='ds5', con=engine, if_exists='replace',
                       index=False, index_label=False)

    d45.to_sql(name='ds45', con=engine, if_exists='replace',
                 index=False, index_label=False)

    d4['process_type'] = None
    d55 = pd.concat([d4, d5])

    d55.to_sql(name='ds55', con=engine, if_exists='replace',
               index=False, index_label=False)


def get_cvs_dataset():
    mask_code_seq = {
        'A': 'B-IND',
        'B': 'B-QT',
        'C': 'B-PST',
        'D': 'B-PS'
    }
    cvs_dataset = sql_interactive.get_sql_cvs_dataset()

    cvs_dataset['label'] = cvs_dataset['label'].apply(
        lambda x: [mask_code_seq[i] if i in mask_code_seq.keys() else 'O' for i in x]
    )


def test_flow_dataset():
    d4, d5 = get_QA_dataset()
    d4 = d4[['industry', 'question_type', 'process']]
    d5 = d5[['industry', 'question_type', 'process_type', 'process']]
    d4.dropna(inplace=True)
    d5.dropna(inplace=True)
    data_template = pd.DataFrame(open(file='../Dataset/测试模板.txt'), columns=['template'])
    data_template['template'] = data_template['template'].apply(
        lambda x: x.strip()
    )

    data_template_list = list(data_template['template'].values)
    templates = []
    labels = []

    mask_code_seq = {
        'A': 'B-IND',
        'B': 'B-QT',
        'C': 'B-PST',
        'D': 'B-PS'
    }

    for dtl in data_template_list:
        l_dtl = str(dtl)
        c_dtl = str(dtl)
        if 'c' not in dtl:
            for i in range(2000):
                d4_ = d4.sample(n=1)
                industry = d4_['industry'].values[0]
                process = d4_['process'].values[0]
                templates.append(c_dtl.replace('a', industry).replace('d', process))
                labels.append(l_dtl.replace('a', str(len(industry) * 'A'))
                              .replace('d', str(len(process) * 'D')))
        else:
            for i in range(2000):
                d5_ = d5.sample(n=1)
                industry = d5_['industry'].values[0]
                process_type = d5_['process_type'].values[0]
                process = d5_['process'].values[0]
                templates.append(c_dtl.replace('a', industry).replace('c', process_type).replace('d', process))

                labels.append(l_dtl.replace('a', len(industry) * 'A')
                              .replace('c', len(process_type) * 'C')
                              .replace('d', len(process) * 'D'))

    templates = pd.DataFrame(np.array(templates).reshape(-1, 1), columns=['content'])
    labels = pd.DataFrame(np.array(labels).reshape(-1, 1), columns=['label'])
    test_data = pd.concat([templates, labels], axis=1)
    test_data['label'] = test_data['label'].apply(
        lambda x: [mask_code_seq[i] if i in mask_code_seq.keys() else 'O' for i in x]
    )

    return test_data
