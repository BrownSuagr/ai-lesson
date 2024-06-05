# 导入模块
import socket

# 1、创建套接字对象
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2、绑定IP与端口号，其参数是一个元组类型的数据
tcp_server_socket.bind(("", 8888))

# 3、设置监听
tcp_server_socket.listen(128)

# 4、等待客户端连接（阻塞）
while True:
    new_socket, ip_port = tcp_server_socket.accept()
    # 5、接收客户端发送过来的消息
    recv_data = new_socket.recv(1024)
    recv_data = recv_data.decode('gbk')
    print(f'{ip_port}客户端发送的消息：{recv_data}')
    # 6、处理请求，把处理结果返回客户端
    content = '信息已收到，over，over！'.encode('gbk')
    new_socket.send(content)
    # new_socket.close()