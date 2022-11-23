#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/5 16:33
# @Author  : Lousix

import base64
import hashlib
import requests
from flask import Flask, request

app = Flask(__name__)

rootCer = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
rootSercretKey = "CUHK"


def encrypt(key, plain):
    origin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    dictionary_decode = str.maketrans(key, origin)
    dictionary_encode = dict(zip(dictionary_decode.values(), dictionary_decode.keys()))
    result_b64 = base64.b64encode(plain.encode()).decode()
    new_result_b64 = result_b64.translate(dictionary_encode)
    return new_result_b64


def decrypt(key, cipher):
    origin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    dictionary_decode = str.maketrans(key, origin)
    new_data = cipher.translate(dictionary_decode)
    result_b64 = base64.b64decode(new_data.encode()).decode()
    return result_b64


@app.route('/')
def step0():
    return 'Step0: CUHK Start Listening...'


@app.route("/step2", methods=['POST', 'GET'])
def step2():
    data = request.get_json()
    pubKey = data['pubKey']
    res = {}
    if data['msg'] == "CSR":
        cer = encrypt(pubKey, rootCer)
        sign = hashlib.md5(cer.encode()).hexdigest()
        print("RECV a CSR from Student:")
        print("Student's pubKey is {}".format(pubKey))
        res = {"code": 200, "msg": "success", 'cer': cer, 'sign': sign}
    else:
        res = {"code": "400", "msg": "Please send a legal CSR!"}
    requests.post("http://127.0.0.1:8889/step2", json=res)
    return res


if __name__ == '__main__':
    app.run('0.0.0.0', 8891)
