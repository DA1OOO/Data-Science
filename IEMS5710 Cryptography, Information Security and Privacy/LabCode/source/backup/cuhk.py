#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/22 19:07
# @Author  : Lousix
import socket

# Step0-2: Listening and Send Cer
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("127.0.0.1", 4444)
serv.bind(server_address)
serv.listen(5)
conn, addr = serv.accept()
from_client = ''
while True:
    data = str(conn.recv(4096), encoding='utf8')  # 接收到的数据类型为byte，转换成str
    if not data:
        break
    from_client = from_client + data
    print(from_client)
    data = open("CA/ca.crt", 'r').read()
    conn.send(bytes(f"{data}", encoding='utf8'))  # 将str转换成byte类型，传送时需要用byte
conn.close()
print("client disconnected")



