# *- coding: utf-8 -*-

# =================================
# time: 2021.6.21
# author: @唐志林
# function: 开关
# =================================

import json
import redis

from sanic import Sanic
from datetime import datetime
from sanic.response import HTTPResponse

from Models.bert_crf import predict
from SlotProcess.slot_process import slot_match, slot2add, slot2tips

slot_tem_address = redis.Redis(host=str('127.0.0.1'), port=int(6379))
slot_tem_address.set(name='slot', value=json.dumps({}))

app = Sanic(__name__)


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


@app.route('/start/<content:string>')
async def deploy(request, content):
    content = content
    print(content)
    stime = datetime.now()
    keywords = predict(content)

    _, information = keywords_process(keywords=keywords)
    if _ == 'tip':
        print(f"不好意思信息缺失关键词, 请按照提示输入下列关键词: \n{information}")
    if _ == 'ans':
        information = information[0]
        print(f"找到你需要的答案了, 内容如下: \n{information}")
    print(f"cost time: {datetime.now() - stime}")
    return HTTPResponse(json.dumps(information))

if __name__ == '__main__':
    app.run(host="0.0.0.0")

# content='你知道铸造行业树脂砂铸造类型中的焊接对于环境有那些危害'
# content = '铸造行业中的焊接有什么坏的影响'
# content = '树脂砂铸造类型是什么'
# deploy()

# 你知道铸监管科造行业树脂铸造傻类型中的焊啥接对于环境有那些危害
