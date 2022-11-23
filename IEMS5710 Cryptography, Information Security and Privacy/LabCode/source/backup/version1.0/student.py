#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/5 16:33
# @Author  : Lousix
import base64
import hashlib

import requests
from flask import Flask, request, g

app = Flask(__name__)

# CUHK的公钥证书
global stuCer

# ID
stuId = "1155180354"

# 这里的非对称加密 用换表的base64替代一下
priKey = "=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
pubKey = "=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

# 会话密钥
sessionKey = ""


def setCer(a):
    f = open("stuCer.cer", 'w')
    f.write(a)

def getCer():
    f = open("stuCer.cer", 'r')
    return f.read().strip("\n")


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
    print(key)
    origin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
    dictionary_decode = str.maketrans(key, origin)
    new_data = cipher.translate(dictionary_decode)
    result_b64 = base64.b64decode(new_data.encode()).decode()
    return result_b64


@app.route('/')
def Step0():
    requests.post("http://127.0.0.1:8891/step2", json={
        "pubKey": pubKey,
        "msg": "CSR"
    })
    return 'Step1: Sending CSR to CUHK...'


@app.route("/step2", methods=["GET", "POST"])
def step2():
    data = request.get_json()
    print(data)
    # 没有获取到证书
    if data['code'] != 200:
        return data['msg']

    sign = data['sign']
    cipher = data['cer']
    # md5 验证证书是否正确
    if sign != hashlib.md5(cipher.encode()).hexdigest():
        return "The Cer had been changed!"
    print("test1")
    # 解密证书
    stuCer = decrypt(priKey, cipher)
    setCer(stuCer)
    cer = hashlib.md5(stuCer.encode()).hexdigest()
    res = {"msg": "check", 'id': stuId, 'cer': cer}
    requests.post("http://127.0.0.1:8890/step3", json=res)
    return res


@app.route("/step5", methods=['POST'])
def step5():
    data = request.get_json()
    print(data)
    sessionKey = decrypt(getCer(), data['sessionKey'])
    print(sessionKey)
    msg = "This is submission from SID {}".format(stuId)
    for i in range(10):
        requests.post("http://127.0.0.1:8890/step6", json={
            "msg": msg,
            "mac": hashlib.md5((sessionKey + msg).encode()).hexdigest()
        })
    return sessionKey


if __name__ == '__main__':
    app.run('0.0.0.0', 8889)
