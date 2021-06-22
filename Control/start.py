# *- coding: utf-8 -*-

# =================================
# time: 2021.6.21
# author: @唐志林
# function: 开关
# =================================

import json
import redis


from SqlLink import sql_interactive
from Models.bert_crf import fit_dataset, predict, get_slot
from SlotProcess.slot_process import slot_match, slot2add, slot2tips

slot_tem_address = redis.Redis(host=str('127.0.0.1'), port=int(6379))
# slot_tem_address.set(name='slot', value=json.dumps({}))

def keywords_process(keywords):
    """
    检查对应插槽的值是否已存在在原数据
    """
    industry = keywords['industry']
    question_type = keywords['question_type']
    process_type = keywords['process_type']
    process = keywords['process']
    answers = []

    slot = slot_tem_address.get(name='slot')
    # 从缓存中提取slot, 如果不存在slot则表明为刚开始进行对话或上一轮对话未出现缺失插槽数据, 返回的是None


    if slot is not None:
        slot = json.loads(slot)
        answers, df, slot = slot2add(industry, question_type,
                                     process, process_type,
                                     answers=answers,
                                     slot=slot)
        slot_tem_address.set(name='slot', value=json.dumps({}))

        return answers

    else:
        slot = {'industry': [], 'process': [], 'question_type': [], 'process_type': []}
        answers, df, slot = slot_match(industry, question_type,
                                       process, process_type,
                                       answers=answers,
                                       slot=slot)

    if len(answers) == 0:
        tips = slot2tips(df, slot)
        if slot:
            slot_tem_address.set(name='slot', value=json.dumps(slot))

        return tips
    else:
        slot_tem_address.set(name='slot', value=json.dumps({}))

        return answers


def deploy():
    # content = input()
    # label = predict(content, crf=True)
    # keywords = get_slot(content=content, predict_label=label)
    keywords = {
        'industry' : ['铸造'],
        'question_type' : [],
        'process' : ['焊接'],
        'process_type' : ['树脂砂铸造类型']
    }

    information = keywords_process(keywords)
    print(information)
    # todo 提供一个接口用以直接获取提示信息 或者重新进行一个问句的输入然后获取相关信息

deploy()
# print(slot_tem_address.get('slot'))


