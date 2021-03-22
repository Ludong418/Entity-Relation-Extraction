import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class LoadModelBase(object):
    def __init__(self, export_dir=None, url=None, model_name='models', signature_name=None):
        """
        预测的基类，分为两种方式预测
        a. grpc 请求方式
        b. 本地导入模型方式
        TODO: 增加http连接方式和 load  model 的方式

        :arg
        url: string类型，用于调用模型测试接口，host:port，例如'10.0.10.69:8500'
        export_dir: string类型，模型本地文件夹目录，r'model\1554373222'
        model_name: string类型，tensorflow serving 启动的时候赋予模型的名称，当
                    url被设置的时候一定要设置。
        signature_name: string类型，tensorflow serving 的签名名称，当
                    url被设置的时候一定要设置。

        :raise
        url和export_dir至少选择一个，当选择url的时候，model_name和signature_name不能为
        None
        """
        if url is None and export_dir is None:
            raise ValueError('`url` or `export_dir`is at least of one !')

        self.export_dir = export_dir
        self.url = url
        self.model_name = model_name
        self.signature_name = signature_name

    def load_pb_model(self):
        predict_fn = tf.contrib.predictor.from_saved_model(self.export_dir)

        return predict_fn

    def load_grpc_connect(self):
        if self.model_name is None or self.signature_name is None:
            raise ValueError('`model_name` and `signature_name` should  not NoneType')

        channel = grpc.insecure_channel(self.url)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name

        return stub, request

    def load_http_connect(self):
        pass

    def load_model(self):
        pass