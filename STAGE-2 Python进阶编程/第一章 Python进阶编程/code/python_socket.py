import socket


def main():
    print('主函数入口')

    # 网络开发的常见协议：TCP协议、UDP协议
    # 套接字的基本概念：socket连接工具

    # TCP网络客户端开发的基本流程
    #     - 创建客户端连接的套接字
    #     - 与服务器套接字建立连接
    #     - 发送数据
    #     - 接收数据
    #     - 关闭客户端套接字

    # TCP服务端开发基本流程：
    #     - 创建服务端套接字对象
    #     - 绑定端口号
    #     - 设置监听
    #     - 等待客户端连接请求
    #     - 接收数据
    #     - 发送数据
    #     - 关闭套接字

    # socket_client()
    socket_server()


def socket_server():
    # 1、创建TCP连接服务器端套接字连接对象， AF_INET = IPV4 SOCK_STEAM = TCP
    tcp_socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 2、绑定连接的端口
    tcp_socket_server.bind(('', 8081))
    # tcp_socket_server.bind(('10.6.174.242', 8081))
    print(tcp_socket_server)

    # 3、设置监听端口和最大连接数量
    tcp_socket_server.listen(128)

    # 4、等待客户端连接（返回对象：第一个参数是新的套接字对象，第二个对象是客户端的信息）
    new_tcp_socket_server, ip_port = tcp_socket_server.accept()
    print(f'服务端：{new_tcp_socket_server}, 端口：{ip_port}')

    # 5、接收客户端发送的数据，将字节流对象转化为字符串格式
    recv_data = new_tcp_socket_server.recv(2048)
    print(f'接收到数据：{recv_data}')
    # recv_data = recv_data.decode('utf-8')

    # 6、处理客户端请求并返回成功数据给客户端
    new_tcp_socket_server.send('已经收到客户端发送的数据'.encode('utf-8'))

    # 7、关闭服务端的套接字对象和接收新产生的套接字对象
    new_tcp_socket_server.close()
    tcp_socket_server.close()


def socket_client():

    # 创建客户端的套接字, socket.AF_INET = TPv4 socket.SOCK_STREAM = TCP
    tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 和服务器套接字建立连接
    tcp_client_socket.connect(('10.6.174.242', 8081))
    # 发送数据
    tcp_client_socket.send('Hello World'.encode(encoding='utf-8'))
    # 接收数据
    recv_data = tcp_client_socket.recv(1024).decode('utf-8')
    print(recv_data)
    # 关闭连接
    tcp_client_socket.close()


if __name__ == '__main__':
    main()
