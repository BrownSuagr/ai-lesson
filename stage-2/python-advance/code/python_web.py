import socket


def main():
    print('主函数入口')

    # 1、创建套接字对象
    # socket.socket(socket_family, socket_type, protocol, fileno)

    # socket_family：套接字中的网络协议，包括
    #   - AF_INET（IPv4网域协议，如TCP与UDP）
    #   - AF_INET6（IPv6）
    #   - AF_UNIX（UNIX网域协议）

    # socket_type：套接字类型，包括
    #   - SOCK_STREAM（使用在TCP中)
    #   - SOCK_DGRAM（使用在UDP中）
    #   - SOCK_RAW（使用在IP中）
    #   - SOCK_SEQPACKET（列表连接模式）

    # protocol：只使用在family等于AF_INET或type等于SOCK_RAW的时候。protocol是一个常数，用于辨识所使用的协议种类。默认值是0，表示适用于所有socket类型。
    # fileno: 套接字的文件描述符
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口可以重用
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    # 2、设置套接字IP和端口
    tcp_server_socket.bind(('10.6.165.0', 8000))
    # tcp_server_socket.bind(('', 8000))

    # 3、设置tcp监听
    tcp_server_socket.listen(128)

    while True:
        # 4、等待客户端连接，返回连接的新socket对象和IP端口号
        new_socket, ip_port = tcp_server_socket.accept()

        # 5、接收客户端消息
        clint_send_data = new_socket.recv(1024).decode('utf-8')
        print(f'客户端数据：{clint_send_data}')

        # 6、发送消息
        with open('./../pythonWeb/index.html', 'rb') as f:
            data = f.read()

        # 6.1、响应行数据
        response_line = 'HTTP/1.1 200 OK\r\n'

        # 6.2、响应头数据
        response_header = 'Server:PythonWeb1.0\r\n'

        # 6.3、 响应体
        response_body = data

        # 6.4、拼接HTTP响应报文
        response_data = (response_line + response_header + '\r\n').encode('utf-8') + response_body

        # 6.5、发送响应数据
        new_socket.send(response_data)

        # new_socket.close()


    # 7、关闭服务端连接
    #     tcp_server_socket.close()


if __name__ == '__main__':
    main()