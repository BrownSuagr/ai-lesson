# 导入socket模块
import socket
# 导入多线程模块
import threading


def socket_accept(parent_socket):
    # 5、接受客户端请求数据
    new_socket, ip_port = parent_socket.accept()
    print(f'new_socket:{new_socket}')
    print(f'ip_port:{ip_port}')

    # 6、接受数据并格式化，接受数据为空，关闭当前套接字
    recv_data = new_socket.recv(2048).decode('utf-8')
    if 0 == len(recv_data):
        new_socket.close()
    else:
        new_socket.send('已经收到数据'.encode('utf-8'))
        return recv_data


def main():
    # 1、创建套接字对象AF_INET IPV4
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    # 2、绑定IP和端口
    tcp_server_socket.bind(('', 9000))

    # 3、设置socket最大监听数量
    tcp_server_socket.listen(128)

    # 4、循环获取客户端发送的请求
    while True:
        socket_thread = threading.Thread(target=socket_accept, kwargs={'parent_socket': tcp_server_socket})
        socket_thread.start()


if __name__ == '__main__':
    main()

