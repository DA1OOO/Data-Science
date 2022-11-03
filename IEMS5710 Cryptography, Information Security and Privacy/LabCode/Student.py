## 客户端
import socket


# 连接到目标端口
def connect_port(port):
    # 创建socket 对象
    my_socket = socket.socket()
    # 连接到本地计算机上的服务器上的port端口
    my_socket.connect(('127.0.0.1', port))
    return my_socket


if __name__ == '__main__':
    my_sokect = connect_port(9335)
    # 从服务器接收数据并解码以获取字符串。
    print(my_sokect.recv(1024).decode())
    my_sokect.close()
