#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/5 20:53
# @Author  : Lousix
import hashlib
import json

import requests
from prettytable import PrettyTable

# 首先是Student 给 CUHK 发自己pubKey和CSR请求
step12 = """SEND to CUHK
pubKey: \"=ABCDE...456789+/\"
msg: \"CSR\""""
r = requests.post("http://127.0.0.1:8891/step2", json={
    "pubKey": "=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
    "msg": "CSR"
})
data = json.loads(r.text)

# CUHK 给 Student 发用Student的pubKey加密的证书cer和不加盐值的md5签名sign
step21 = f"""SEND to Student
cer: {data['cer'][:5]}...{data['cer'][-5:]}
sign: {data['sign']}"""
r = requests.post("http://127.0.0.1:8889/step2", json=data)
data = json.loads(r.text)

# Student 用 md5 验证证书的正确性，再解密
# Student 给 BlackBoard 发用md5加密的cer和学号id
step32 = f"""SEND to BlackBoard
cer: {data['cer']}
id: {data['id']}"""
r = requests.post("http://127.0.0.1:8890/step3", json=data)
data = json.loads(r.text)

# BlackBoard 用 md5 验证证书的正确性
# BlackBoard 给 Student 发用证书加密的会话密钥sessionKey
step43 = f"""SEND to Student
sessionKey: {data['sessionKey']}"""
r = requests.post("http://127.0.0.1:8889/step5", json=data)
data = r.text
step52 = f"""GET SessionKey
sessionKey: {data}
"""

# Student 解密 sessionKey
# 把 sessionKey 作为 salt 进行 md5， 发送msg
stuId = "1155180354"
msg = "This is submission from SID {}".format(stuId)
step62 = f"""SEND to BlackBoard (x10)
msg: {msg}
mac: {hashlib.md5((data + msg).encode()).hexdigest()}"""
r = requests.post("http://127.0.0.1:8890/step6", json={
    "msg": msg,
    "mac": hashlib.md5((data + msg).encode()).hexdigest()
})

# BlackBoard 验证
step73 = f"""CHECK
{r.text}"""

# 画表格
table = PrettyTable(['step', 'CUHK', 'Student', 'BlackBoard'])
table.add_row(['Step0', 'Listening...', '', 'Listening...'])
table.add_row(['Step1', '', f'{step12}', ''])
table.add_row(['Step2', f'{step21}', '', ''])
table.add_row(['Step3', '', f'{step32}', ''])
table.add_row(['Step4', '', '', f'{step43}'])
table.add_row(['Step5', '', f'{step52}', ''])
table.add_row(['Step6', '', f'{step62}', ''])
table.add_row(['Step7', '', '', f'{step73}'])
table.align = 'l'
print(table)
