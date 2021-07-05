# *- coding: utf-8 -*-

# =================================
# time: 2021.6.28
# author: @唐志林
# function: 部署相关
# =================================

import grpc
import tensorflow as tf
from datetime import datetime

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from Models.bert_crf import get_predict_data, get_keywords_label, get_slot
from start import keywords_process


def data2url(input_data, server_url):
    """
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param input_data: ids, masks, tokens, target
    :param server_url: TensorFlow Serving的地址加端口: localhost:port
    """
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "test_tfs"  # 模型名称，启动容器命令的model_name参数
    request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的

    ids, masks, token_types_ids, target = input_data


    request.inputs["ids"].CopyFrom(tf.make_tensor_proto(ids))

    request.inputs["mask"].CopyFrom(tf.make_tensor_proto(masks))

    request.inputs["tokens"].CopyFrom(tf.make_tensor_proto(token_types_ids))

    request.inputs['target'].CopyFrom(tf.make_tensor_proto(target))


    response = stub.Predict(request)  # 5 secs timeout

    predict = tf.constant(response.outputs['output_1'].int_val)

    return predict


def deployment_predict(url):
    """1: 将输入的问句进行初步token处理---get_predict_data(）
    2: 将数据发送到模型部署的地址url
    3: 将序列标注结果进行处理获取关键词和关键词对应槽值---get_keywords_label()
    4: 将关键词插入到对应的槽里面---get_slot()
    """
    content = input("请输入相关信息: ")
    stime = datetime.now()
    input_data, label_length =  get_predict_data(content)
    predict_label = data2url(input_data=input_data, server_url=url)
    predict_label = tf.expand_dims(predict_label, 0)

    keywords, predict_label = get_keywords_label(predict_label=predict_label,
                                                 label_length=label_length,
                                                 input_id=input_data[0])

    keywords = get_slot(keywords, predict_label)
    _, information = keywords_process(keywords=keywords)
    print("")
    if _ == 'tip':
        print(f"不好意思信息缺失关键词, 请按照提示输入下列关键词: \n{information}")
    if _ == 'ans':
        information = information[0]
        print(f"找到你需要的答案了, 内容如下: \n{information}")
    print(f"cost time: {datetime.now() - stime}")
    return deployment_predict(url)

deployment_predict(url='localhost:8500')
# content='你知道铸造行业树脂砂铸造类型中的焊接对于环境有那些危害'
# content = '铸造行业中的焊接有什么坏的影响'
# content = '树脂砂铸造类型是什么'