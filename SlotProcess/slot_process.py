# *- coding: utf-8 -*-

# =================================
# time: 2021.6.04
# author: @唐志林
# function: 槽处理
# =================================

import pandas as pd

from SqlLink import sql_interactive
from Models.similarity_process import get_Top1similarity_word

pd.set_option('display.max_columns', None)


def slot_match(industry, question_type, process, process_type, answers, slot):
    """使用该方法进行槽数据的匹配, 然后索引到对应的答案
    """
    answer_dataset = sql_interactive.get_sql_QA_dataset(dataset_name='ds55')
    # 对于max_count>1的数据逻辑比较复杂
    cou_ind = len(industry)
    cou_qt = len(question_type)
    cou_p = len(process)
    cou_pt = len(process_type)

    def process_answers():
        # 用于处理answers内部的可能出现的Dataframe
        df = None
        for num, ans in enumerate(answers):
            if type(ans) != str:
                df = answers.pop(num)

        return answers, df, slot

    def pop_keywords4str(data):
        # 当得到了一个字符串式的answer调用该方法
        # 用以将对应的ind和pt以及p列表内的已经使用的索引关键词清楚
        ind_ = data['industry'].values[0]
        p_ = data['process'].values[0]
        pt_ = data['process_type'].values[0]

        if ind_ in industry:
            industry.pop(industry.index(ind_))
        if p_ in process:
            process.pop(process.index(p_))
        if pt_ in process_type:
            process_type.pop(process_type.index(pt_))

    def pop_keywords4df(data):
        # 当索引的answer并不是字符串而是Dataframe时
        # 用以处理对应ind以及p和pt列表内的索引关键词
        ind4data = data['industry'].values
        p4data = data['process'].values
        pt4data = data['process_type'].values


        for ind4, p4, pt4 in zip(ind4data, p4data, pt4data):
            if ind4 in industry:
                slot['industry'].append(industry.pop(industry.index(ind4)))
            if p4 in process:
                slot['process'].append(process.pop(process.index(p4)))
            if pt4 in process_type:
                slot['process_type'].append(process_type.pop(process_type.index(pt4)))

    if max(cou_ind, cou_qt, cou_p, cou_pt) == 0:
        return process_answers()

    if cou_p > 0:
        p = process[0]
        answer_dataset_p = answer_dataset.loc[answer_dataset['process'] == p]
        if len(answer_dataset_p) == 1:
            answer = answer_dataset_p['answer'].values[0]
            ind = answer_dataset_p['industry'].values[0]
            pop_keywords4str(answer_dataset_p)
            answers.append(f"{ind}-{p}: {answer}")
            return slot_match(industry, question_type, process, process_type, answers, slot)
        else:
            answer_dataset = answer_dataset_p

    if cou_pt > 0:
        for pt in process_type:
            if pt in answer_dataset['process_type'].values:
                answer_dataset_pt = answer_dataset.loc[answer_dataset['process_type'] == pt]
                # 删掉pt列表中对应的值
                if len(answer_dataset_pt) == 1:
                    answer = answer_dataset_pt['answer'].values[0]
                    ind = answer_dataset_pt['industry'].values[0]
                    p = answer_dataset_pt['process'].values[0]
                    pop_keywords4str(answer_dataset_pt)
                    answers.append(f"{ind}-{pt}-{p}: {answer}")
                    return slot_match(industry, question_type, process, process_type, answers, slot)
                else:
                    answer_dataset = answer_dataset_pt

    if cou_ind > 0:
        for ind in industry:
            if ind in answer_dataset['industry'].values:
                answer_dataset_ind = answer_dataset.loc[answer_dataset['industry'] == ind]
                # 删掉ind列表中对应的值
                if len(answer_dataset_ind) == 1:
                    answer = answer_dataset_ind['answer'].values[0]
                    p = answer_dataset_ind['process'].values[0]
                    pop_keywords4str(answer_dataset_ind)
                    answers.append(f"{ind}-{p}: {answer}")
                    return slot_match(industry, question_type, process, process_type, answers, slot)
                else:
                    answer_dataset = answer_dataset_ind

    answer = answer_dataset
    pop_keywords4df(answer)
    answers.append(answer)

    if max(cou_ind, cou_qt, cou_p, cou_pt) > 0:
        return slot_match(industry, question_type, process, process_type, answers, slot)


def slot2tips(df, slot):
    """假如当前槽数据不够完整, 那么通过此方法找到缺失的关键词数据是什么
    """
    df.drop(['answer', 'question_type'], axis=1, inplace=True)
    df_col = df.columns

    tip_inf = {}
    for col in df_col:
        if col not in slot or len(slot[col]) == 0:
            tip_inf[col] = list(set(df[col].values))

    return tip_inf


def slot2add(industry, question_type, process, process_type, answers, slot):
    """将现阶段获取的槽数据与上个阶段遗留的槽数据结合到一起, 重新进行槽数据的匹配进行答案的查找
    """
    for ind in industry:
        slot['industry'].append(ind)
    for qt in question_type:
        slot['question_type'].append(qt)
    for p in process:
        slot['process'].append(p)
    for pt in process_type:
        slot['process_type'].append(pt)

    industry = slot['industry']
    question_type = slot['question_type']
    process_type = slot['process_type']
    process = slot['process']

    slot = {
        'industry': [],
        'process': [],
        'question_type': [],
        'process_type': []
    }

    answers, df, slot = slot_match(industry, question_type, process, process_type, answers=answers, slot=slot)

    return answers, df, slot


def check_slot(keywords):
    """该函数的目的用于检查获取的槽数据是否正确
    如果正确则返回槽数据如果不正确则需要对槽数据里错误的槽进行相似度匹配
    example：
    传递过来的关键词: {'industry': ['铸造'], 'question_type': [], 'process_type': [], 'process': ['焊接']}
    则该函数会检查'铸造'是否在槽‘industry’内部,‘焊接’是否在槽‘process‘内部如果均在则返回正确否则
    返回错误，会进行下一步的关键相似度匹配计算找到最合适的关键词替换到对应的槽内部
    """
    print(f"包含错误关键词的槽数据: {keywords}")
    new_keywords = {}
    for key, values in keywords.items():
        new_keywords[key] = []
        if len(values) >= 1:
            check_key4values = sql_interactive.get_sql_check_slot_dataset(slot=key)
            # 从数据库里面获取到对应槽的去重值--返回一个列表
            for v in values:
                if v in check_key4values:
                    pass
                else:
                    v = get_Top1similarity_word(word=v, similarity_words=check_key4values)
                new_keywords[key].append(v)
    print(f"最终获取的正确关键词: {new_keywords}")
    return new_keywords

