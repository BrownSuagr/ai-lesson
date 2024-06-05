# 导入模块
import socket

# 1、创建套接字对象
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 2、绑定IP和端口
tcp_server_socket.bind(("", 9999))
# 3、设置监听
tcp_server_socket.listen(128)
while True:
    # 4、等待客户端连接
    new_socket, ip_port = tcp_server_socket.accept()
    # print(ip_port)
    # 5、接收客户端发送过来的请求（http结构请求报文）
    recv_data = new_socket.recv(1024)
    # 过滤掉浏览器的异常请求（浏览器关闭后，不再接收任何消息）
    if recv_data:
        recv_data = recv_data.decode('utf-8')
        # 把recv_data使用split进行切割（maxsplit参数代表最多切几刀）
        request_list = recv_data.split(' ', maxsplit=2)
        # 获取用户想要访问的资源路径
        request_path = request_list[1]  # /gdp.html

        # 判断用户请求的request_path路径是否为斜杠/
        if request_path == '/':
            request_path = '/index.html'
        # source/html/index.html => 如果用户没有输入访问路径，默认返回这个项目的首页index.html

        # 根据请求资源路径返回指定页面的数据（http响应报文）
        # 'source/html' + '/gdp.html' == source/html/gdp.html
        try:
            with open('source/html' + request_path, 'rb') as f:
                data = f.read()
        except:
            # 代表文件未找到，把HTTP响应报文设置为404
            response_line = 'HTTP/1.1 404 Not Found\r\n'
            response_header = 'Server:PWB1.0\r\nContent-type:text/html; charset=utf-8\r\n'
            response_body = '很抱歉，您要访问的资源不存在，正在努力维护中...\r\n'
            response_data = (response_line + response_header + '\r\n' + response_body).encode('utf-8')

            # 返回数据
            new_socket.send(response_data)
        else:
            # 代表文件找到了，把HTTP响应报文设置为200
            response_line = 'HTTP/1.1 200 OK\r\n'
            response_header = 'Server:PWB1.0\r\n'
            response_body = data

            response_data = (response_line + response_header + '\r\n').encode('utf-8') + response_body

            # 返回数据给客户端浏览器
            new_socket.send(response_data)
        finally:
            # 关闭套接字对象
            new_socket.close()