## 客户端
import socket

# 创建socket 对象
s = socket.socket()
# 定义要连接的服务端端口
port = 80
# 连接到本地计算机上的服务器上的port端口
s.connect(('127.0.0.1', port))
# 从服务器接收数据并解码以获取字符串。
print(s.recv(1024).decode())
# 关闭连接
s.close()
