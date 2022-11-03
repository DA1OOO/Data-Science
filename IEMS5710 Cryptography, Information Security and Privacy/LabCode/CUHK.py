# CUHK服务端 9335端口
import socket


# 初始化Socket
def initial_socket():
    # 创建套接字
    try:
        my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Create socket succeeded!")
    except socket.error as err:
        print("Create socket failed!，error details: %s" % (err))
    # 套接字默认端口
    port = 9335
    # 监听本机上80端口的网络请求
    my_socket.bind(('127.0.0.1', port))
    # 切换套接字到监听模式，最多阻塞5笔请求
    my_socket.listen(5)
    print("Socket listening...")
    return my_socket


# 接受客户端连接，并保持监听
def connect_accept(my_socket):
    while True:
        # 与客户端建立连接。
        c, addr = my_socket.accept()
        print('Connect Success!', addr)
        # 向客户发送感谢信息。编码以发送字节类型。
        c.send('Thanks for your connect'.encode())
        # 关闭与客户端的连接
        c.close()


if __name__ == '__main__':
    my_socket = initial_socket()
    connect_accept(my_socket)
