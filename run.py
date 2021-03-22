#!/usr/bin/python

# encoding: utf-8

"""
@author: dong.lu

@contact: ludong@cetccity.com

@software: PyCharm

@file: infer.py

@time: 2019/04/9 10:30

@desc: 实体提取服务

"""
import json

import tensorflow as tf
from flask import Flask, abort, Response
from flask_cors import *
from flask_restful import reqparse, Api, Resource

from combinative_infer import Infer


app = Flask(__name__)
CORS(app, supports_credentials=True)
app.app_context().push()
api = Api(app)

# 导入 tensorflow 默认计算图
graph = tf.get_default_graph()

infer = Infer()


class PredicateExtract(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('sentences', type=str, action='append')

    def post(self):
        args = self.parser.parse_args()
        sentences = args.get('sentences')
        response_data = dict(labels=None, success=False, model_name='PredicateExtract')
        if not sentences:
            res = Response(json.dumps({'msg': 'The sentence cannot be empty ', 'success': False}), status=400)
            return abort(res)

        try:
            result = infer.infer(sentences)
            response_data.update(labels=result)
            response_data.update(success=True)
            return response_data, 200

        except:
            res = Response(json.dumps({'msg': 'The sentence cannot be classify ', 'success': False}), status=500)
            return abort(res)


api.add_resource(PredicateExtract, '/modelapp/api/v0/predicateExtract')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)