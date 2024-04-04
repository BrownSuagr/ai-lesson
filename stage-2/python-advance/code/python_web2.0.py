# 导入模块
import socket


# 定义主函数入口
def main():

    # 1、创建一个TCP的socket连接对象
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置socket端口可以复用
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    # 2、设置socket连接IP和端口
    tcp_server_socket.bind(('', 8000))

    # 3、设置socket监听数量
    tcp_server_socket.listen(128)

    # 4、循环等待客户端连接
    while True:

        # 4.1、同意接收并获取客户端连接和端口号
        new_socket_conn, ip_port = tcp_server_socket.accept()
        print(f'客户端连接对象：{new_socket_conn}')
        print(f'客户端连接地址端口：{ip_port}')

        # 5、通过新连接获取客户端请求字节流数据，设置接收数据长度和数据格式
        receive_data = new_socket_conn.recv(1024).decode('utf-8')
        print(f'客户端请求数据：{receive_data}')

        # 5.1、判断客户端是否有数据连接，否怎断开
        # if not receive_data:
        #     new_socket_conn.close()
        #     break

        # 获取请求的路由地址
        router = receive_data.split(' ', maxsplit=2)
        print(f'路由地址:{router}')

        # 6、组装需要发送的请求数据
        response_line = 'HTTP/1.1 200 OK\r\n'
        response_header = 'Server:PythonWeb2.0\r\n'
        # with open('./../pythonWeb/index.html', 'rb') as file:
        with open('./../pythonWeb' + router[1], 'rb') as file:
            file_data = file.read()
        response_data = file_data
        response_data = (response_line + response_header + '\n\r').encode('utf-8') + response_data

        # 7、发送数据
        new_socket_conn.send(response_data)


if __name__ == '__main__':
    main()


