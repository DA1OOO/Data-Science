## 服务端
import socket

# 创建套接字
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("已成功创建套接字")
except socket.error as err:
    print("套接字创建失败，出现错误 %s" % (err))
# 套接字默认端口
port = 80
# 监听本机上80端口的网络请求
s.bind(('127.0.0.1', port))
# 切换套接字到监听模式，最多阻塞5比请求
s.listen(5)
print("套接字正在侦听")
# 一个永远的循环，直到我们中断它或发生错误
while True:
    # 与客户端建立连接。
    c, addr = s.accept()
    print('已从获得连接', addr)
    # 向客户发送感谢信息。编码以发送字节类型。
    c.send('感谢您的连接'.encode())
    # 关闭与客户端的连接
    c.close()
    # 打破循环
    break
