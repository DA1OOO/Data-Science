#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/22 19:08
# @Author  : Lousix
import hashlib
import socket
import ssl

# Step0-2 明文请求CSR, 保存在本地client中
SID = "WHY"
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("127.0.0.1", 4444)
client.connect(server_address)
client.send(bytes(f"I am {SID}\n", encoding='utf8'))
from_server = str(client.recv(4096), encoding='utf8')
print(from_server)
f = open("client/CA.crt", 'w')
f.write(from_server)
f.close()
client.close()



# Socket初始化
context = ssl.SSLContext(ssl.PROTOCOL_TLS)
context.check_hostname = False
context.load_verify_locations("./client/CA.crt")
context.load_cert_chain('./client/client.crt', './client/client.key')
context.verify_mode = ssl.CERT_REQUIRED
sock = socket.create_connection(('127.0.0.1', 4443))


# Step3 校验一下, 我觉得我这里写的很奇怪,
# 逻辑是 只要B能够解析出请求里有check, 就算是成功,
# 因为如果解析不出来就说明CA证书有问题
ssock = context.wrap_socket(sock, server_side=False)
ssock.send(bytes("check I am client\n", encoding='utf8'))
sessionKey = str(ssock.recv(4096), encoding='utf8')
print(sessionKey)
ssock.close()


# Step4-6 拿到sessionKey
# 签名 拼接 发送
req = "This is submission from SID {}".format(SID)
for i in range(10):
    sock = socket.create_connection(('127.0.0.1', 4443))
    ssock = context.wrap_socket(sock, server_side=False)
    msg = f"This is submission from SID {SID}#{str(hashlib.md5((sessionKey + req).encode()).hexdigest())}"
    ssock.send(bytes(msg, encoding='utf8'))
    response = str(ssock.recv(4096), encoding='utf8')
    print(response)

ssock.close()
