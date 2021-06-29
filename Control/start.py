# *- coding: utf-8 -*-

# =================================
# time: 2021.6.21
# author: @唐志林
# function: 开关
# =================================

import json
import redis

from Models.bert_crf import predict, get_slot
from SlotProcess.slot_process import slot_match, slot2add, slot2tips

slot_tem_address = redis.Redis(host=str('127.0.0.1'), port=int(6379))
slot_tem_address.set(name='slot', value=json.dumps({}))

def keywords_process(keywords):
    """
    检查对应插槽的值是否已存在在原数据
    """
    industry = keywords['industry']
    question_type = keywords['question_type']
    process_type = keywords['process_type']
    process = keywords['process']
    answers = []

    slot = json.loads(slot_tem_address.get(name='slot'))
    # 从缓存中提取slot, 如果不存在slot则表明为刚开始进行对话或上一轮对话未出现缺失插槽数据, 返回的是None

    if slot:
        answers, df, slot = slot2add(industry, question_type,
                                     process, process_type,
                                     answers=answers,
                                     slot=slot)
        slot_tem_address.set(name='slot', value=json.dumps({}))

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

        return 'tip', tips
    else:
        slot_tem_address.set(name='slot', value=json.dumps({}))
        return 'ans', answers


def deploy():
    content = input("请输入你想了解的信息: ")
    keywords, label = predict(content, crf=True)
    keywords = get_slot(keywords=keywords, predict_label=label)

    _, information = keywords_process(keywords=keywords)
    if _ == 'tip':
        print(f"不好意思信息缺失关键词, 请按照提示输入下列关键词: \n{information}")
    if _ == 'ans':
        print(f"找到你需要的答案了, 内容如下: \n{information[0]}")
    return deploy()
    # todo 提供一个接口用以直接获取提示信息 或者重新进行一个问句的输入然后获取相关信息


# content='你知道铸造行业树脂砂铸造类型中的焊接对于环境有那些危害'
# content = '铸造行业中关于焊接有什么坏的影响'
# content = '树脂砂铸造类型是什么'
# deploy()
#
# sudo nvidia-docker run -p 8500:8500 -v /mnt/tang_cvs/ModelCkpt/model_save/bert_crf_checkpoint/:/models/test_tfs --name test_tfs 50588334dfbf --port=8500 --per_process_gpu_memory_fraction=0.99 --enable_batching=true --model_name=test_tfs --model_base_path=/models/test_tfs bash &
# nvidia-docker run -p 8500:8500 -v /mnt/tang_cvs/ModelCkpt/model_save/bert_crf_checkpoint/:/models/test_tfs 50588334dfbf --model_name=test_tfs --model_base_path=/models/test_tfs bash &
#
# docker run -it -p 8501:8501 \
#   --mount type=bind,source=/mnt/tang_cvs/ModelCkpt/model_save/bert_crf_checkpoint/,target=/models/bert_crf_checkpoint \
#   -e MODEL_NAME=bert_crf_checkpoint -t 50588334dfbf bash
#
# tensorflow_model_server --port=8500 --rest_api_port=8501 \
#   --model_name=bert_crf_checkpoint --model_base_path=/models/bert_crf_checkpoint
