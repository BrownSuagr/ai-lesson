# 导入模块
import socket

# 创建一个WebServer服务器端类
class WebServer(object):
    # 定义__init__()魔术方法，用于对象初始化
    def __init__(self):
        # 创建套接字对象
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定IP与端口号
        self.tcp_server_socket.bind(("", 8000))
        # 设置监听
        self.tcp_server_socket.listen(128)

    # 定义一个handle_client_request()
    def handle_client_request(self, new_socket, ip_port):
        # 接收客户端消息
        recv_data = new_socket.recv(1024)
        recv_data = recv_data.decode('gbk')
        print(f'{ip_port}客户端发送过来的消息：{recv_data}')

        # 返回（发送）消息给客户端
        content = '信息已收到，over，over！'.encode('gbk')
        new_socket.send(content)

        # 处理完成后，关闭新套接字对象
        # new_socket.close()


    # 定义一个start方法，用于启动WebServer，接收客户端连接
    def start(self):
        while True:
            # 等待客户端连接
            new_socket, ip_port = self.tcp_server_socket.accept()
            # 定义一个方法，用于接收和发送消息
            self.handle_client_request(new_socket, ip_port)


# 创建程序执行入口，实例化WebServer类生成对象
if __name__ == '__main__':
    # 实例化对象
    ws = WebServer()
    # 调用自身方法，用于启动服务
    ws.start()