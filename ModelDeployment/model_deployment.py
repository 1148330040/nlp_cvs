# *- coding: utf-8 -*-

# =================================
# time: 2021.6.28
# author: @唐志林
# function: 部署相关
# =================================

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def request_server(batch_data, server_url):
    """
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param batch_data: ids, masks, tokens, target
    :param server_url: TensorFlow Serving的地址加端口: localhost:port
    """
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "half_plus_two"  # 模型名称，启动容器命令的model_name参数
    request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的

    ids, masks, token_types_ids, target = batch_data


    request.inputs["ids"].CopyFrom(tf.make_tensor_proto(ids))

    request.inputs["mask"].CopyFrom(tf.make_tensor_proto(masks))

    request.inputs["tokens"].CopyFrom(tf.make_tensor_proto(token_types_ids))

    request.inputs['target'].CopyFrom(tf.make_tensor_proto(target))


    response = stub.Predict(request)  # 5 secs timeout

    predict = tf.constant(response.outputs['output_1'].int_val)

    return predict


print(request_server(batch_data=None, server_url='localhost:8500'))