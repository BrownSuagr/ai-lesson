# 导入模块
import socket

# 1、创建套接字对象
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置端口复用
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

# 2、绑定IP与端口
tcp_server_socket.bind(("", 8000))

# 3、设置监听
tcp_server_socket.listen(128)

while True:
    # 4、等待客户端连接（浏览器连接）
    new_socket, ip_port = tcp_server_socket.accept()

    # 5、接收消息（返回结果是一个HTTP请求报文）
    client_request_data = new_socket.recv(1024).decode('utf-8')
    print(client_request_data)

    # 6、发送消息（返回结果是一个HTTP响应报文） => 必须要把数据组装，组装成HTTP响应报文结构
    with open('source/html/gdp.html', 'rb') as f:
        data = f.read()
    # ① 响应行
    response_line = 'HTTP/1.1 200 OK\r\n'
    # ② 响应头
    response_header = 'Server:PythonWeb1.0\r\n'
    # ③ 空行
    # ④ 响应体
    response_body = data
    # 拼接成HTTP响应报文
    response_data = (response_line + response_header + '\r\n').encode('utf-8') + response_body

    new_socket.send(response_data)