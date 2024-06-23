# 导入模块
import socket

# 1、创建套接字对象
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置端口复用，如果程序执行结束，让其占用的端口可以立即释放
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
# 2、绑定IP与端口
tcp_server_socket.bind(("", 8090))
# 3、设置监听
tcp_server_socket.listen(128)
# 4、等待客户端连接
while True:
    # 使用try...except捕获连接异常
    try:
        new_socket, ip_port = tcp_server_socket.accept()
        while True:
            try:
                # 5、接收客户端发送过来的消息
                recv_data = new_socket.recv(1024)

                recv_data = recv_data.decode('gbk')
                print(f'{ip_port}：{recv_data}')

                content = input('服务器端消息：').encode('gbk')
                new_socket.send(content)

            except ConnectionResetError:
                print(f'{ip_port}客户端连接已经断开')
                break
    except:
        print('出错，退出服务器监听')
        break

# 关闭套接字对象
tcp_server_socket.close()
