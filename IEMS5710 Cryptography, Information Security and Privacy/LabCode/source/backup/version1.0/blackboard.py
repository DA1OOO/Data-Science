#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/5 16:34
# @Author  : Lousix
import base64
import hashlib

import requests
from flask import Flask, request

app = Flask(__name__)

rootCer = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/="
sessionKey = "hashsalt"


# 换表的base64 加密算法
def encrypt(key, plain):
    origin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    dictionary_decode = str.maketrans(key, origin)
    dictionary_encode = dict(zip(dictionary_decode.values(), dictionary_decode.keys()))
    result_b64 = base64.b64encode(plain.encode()).decode()
    new_result_b64 = result_b64.translate(dictionary_encode)
    return new_result_b64


# 换表的base64 解密算法
def decrypt(key, cipher):
    origin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    dictionary_decode = str.maketrans(key, origin)
    new_data = cipher.translate(dictionary_decode)
    result_b64 = base64.b64decode(new_data.encode()).decode()
    return result_b64


@app.route('/')
def Step0():
    return 'Step0: BlackBoard Start Listening...'


@app.route("/step3", methods=["GET", "POST"])
def step3():
    data = request.get_json()

    cer = data['cer']
    if hashlib.md5(rootCer.encode()).hexdigest() != cer:
        return "The cer is not right!"

    cipher = encrypt(rootCer, sessionKey)

    res = {"code": 200, "sessionKey": cipher}
    requests.post("http://127.0.0.1:8889/step5", json=res)
    return res


@app.route("/step6", methods=["GET", "POST"])
def step6():
    data = request.get_json()
    msg = data['msg']
    mac = data['mac']
    if hashlib.md5((sessionKey+msg).encode()).hexdigest() == mac:
        print("Message \"{}\" is correct!".format(msg))
        return "Message \"{}\" is correct!".format(msg)


if __name__ == '__main__':
    app.run('0.0.0.0', 8890)
