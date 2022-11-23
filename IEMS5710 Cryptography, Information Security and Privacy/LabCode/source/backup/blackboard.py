#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2022/11/22 19:07
# @Author  : Lousix
import hashlib
import ssl
import socket

ssl._create_default_https_context = ssl._create_unverified_context
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_verify_locations("./CA/ca.crt")
context.load_cert_chain('./server/server.crt', './server/server.key')
context.verify_mode = ssl.CERT_REQUIRED

with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
    with context.wrap_socket(sock, server_side=True) as ssock:
        ssock.bind(('127.0.0.1', 4443))
        ssock.listen(5)
        while True:
            client_socket, addr = ssock.accept()
            msg = client_socket.recv(1024).decode("utf-8")
            print(f"receive msg from client {addr}：{msg}")
            sessionKey = "123"
            response = ""
            if msg.startswith("check"):
                aaa = msg.split(" ")
                response = f"{sessionKey}"
                print(response)
            else:
                aaa = msg.split("#")
                if hashlib.md5((sessionKey + aaa[0]).encode()).hexdigest() == aaa[1]:
                    response = "correct"
            print(response)
            response = response.encode("utf-8")
            client_socket.send(response)
            client_socket.close()