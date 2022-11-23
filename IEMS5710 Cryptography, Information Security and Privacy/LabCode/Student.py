## Student客户端
import socket


# 连接到目标端口
def connect_port(port):
    # 创建socket 对象
    my_socket = socket.socket()
    # 连接到本地计算机上的服务器上的port端口
    my_socket.connect(('127.0.0.1', port))
    return my_socket


# 用户输入
def input_info():
    sid = input("===> Enter your student ID: ")
    # 数据校验
    while len(sid) != 10 or (sid.isdecimal() is not True):
        sid = input("===> Error input! Please input again: ")
    return sid


if __name__ == '__main__':
    my_sokect = connect_port(9335)
    sid = '1234567890'
    msg = str(f"I am {sid}\n")
    my_sokect.send(msg.encode())
    # 从服务器接收数据并解码以获取字符串
    print(my_sokect.recv(1024).decode())
    my_sokect.close()
