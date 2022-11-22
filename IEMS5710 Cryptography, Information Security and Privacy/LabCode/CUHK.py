# CUHK服务端 9335端口
import socket
from OpenSSL import crypto
from OpenSSL import SSL

CERT_FILE = "cuhk.cer"
KEY_FILE = "cuhk.key"


# 初始化Socket
def initial_socket():
    print("-------- Socket Initial -----------")
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
        print("----------------------------------")
        print('Connect Success!', addr)
        # 向客户发送感谢信息。编码以发送字节类型。
        c.send('Thanks for your connect'.encode())
        str = c.recv(1024)
        print("Received msg: %s" % str)
        # 关闭与客户端的连接
        c.close()


def generate_root_ca():
    print("------ Generate self-sign cr ------")
    # create a key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 1024)
    # create a self-signed cert
    cert = crypto.X509()
    cert.get_subject().C = "CN"
    cert.get_subject().ST = "HongKong"
    cert.get_subject().L = "HongKong"
    cert.get_subject().O = "CUHK"
    cert.get_subject().OU = "CUHK-DA1YAYUAN"
    cert.get_subject().CN = "localhost"
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha1')

    # open(join(cert_dir, CERT_FILE), "wt").write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(CERT_FILE, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

    # open(join(cert_dir, KEY_FILE), "wt").write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    with open(KEY_FILE, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    print(" .cer filepath: /%s " % CERT_FILE)
    print(" .key filepath: /%s " % KEY_FILE)
    print("---------- Generate over ----------")


if __name__ == '__main__':
    my_socket = initial_socket()
    generate_root_ca()
    connect_accept(my_socket)
