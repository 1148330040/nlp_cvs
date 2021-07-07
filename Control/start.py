# *- coding: utf-8 -*-

# =================================
# time: 2021.6.21
# author: @唐志林
# function: 开关
# =================================

import os
import json
import redis

from sanic import Sanic
from datetime import datetime
from sanic.response import HTTPResponse

from Models.bert_crf import predict
from SlotProcess.slot_process import slot2match, slot2add, slot2tips

slot_tem_address = redis.Redis(host=str('127.0.0.1'), port=int(6379))
slot_tem_address.set(name='slot', value=json.dumps({}))

app = Sanic(__name__)

slot_log_dir = f'../ModelCkpt/slot_logs/'

if not os.path.exists(slot_log_dir):
    os.makedirs(slot_log_dir)


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
        answers, df, slot = slot2add(new_slot=keywords,
                                     old_slot=slot,
                                     answers=answers)
        # 考虑上下文信息, 在一个阶段结束后不在将缓存置为空
        slot_tem_address.set(name='slot', value=json.dumps(slot))
        with open(slot_log_dir + 'set_slot_logs.txt', 'a') as f:
            # 保留缓存机制内部更改的slot的痕迹
            # 该痕迹文件可用于帮助记录错误和日志以及判断用户
            log = f"time: , user_id: , set the slot to null"
            # f.write(log)

    else:
        slot = {'industry': [], 'process': [], 'question_type': [], 'process_type': []}
        answers, df, slot = slot2match(industry, question_type,
                                       process, process_type,
                                       answers=answers,
                                       slot=slot)

    if len(answers) == 0:
        # answer=0说明没有获取到对应答案
        # 因此调用slot2tips获取对应的提示信息
        tips = slot2tips(df, slot)
        if slot:
            # 因为answer=0说明现有的slot不足以支持获取到answer
            # 因此将现有的slot保存到缓存中
            slot_tem_address.set(name='slot', value=json.dumps(slot))
            with open(slot_log_dir + 'set_slot_logs.txt', 'a') as f:
                # 保留缓存机制内部更改的slot的痕迹
                # 该痕迹文件可用于帮助记录错误和日志以及判断用户
                log = f"time: , user_id: , slot_data: {slot} \n"
                # f.write(log)
        return 'tip', tips
    else:
        # answer不为0则说明现有的slot足以获取到答案因此返回answer
        # slot_tem_address.set(name='slot', value=json.dumps({}))
        return 'ans', answers


@app.route('/start/<content:string>')
async def deploy(request, content):

    content = content
    time = datetime.now()
    keywords = predict(content)

    _, information = keywords_process(keywords=keywords)

    if _ == 'tip':
        print(f"不好意思信息缺失关键词, 请按照提示输入下列关键词: \n{information}")
    if _ == 'ans':
        information = information[0]
        print(f"找到你需要的答案了, 内容如下: \n{information}")
    print(f"cost time: {datetime.now() - time}")
    return HTTPResponse(json.dumps(information))


if __name__ == '__main__':
    app.run(host="0.0.0.0")

# content='你知道铸造行业树脂砂铸造类型中的焊接对于环境有那些危害'
# content = '铸造行业中的焊接有什么坏的影响'
# content = '树脂砂铸造类型是什么'
