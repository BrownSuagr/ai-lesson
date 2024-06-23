# 1、导入模块
import socket
import threading

# 7、定义一个函数，专门用于处理收发消息
def handle_client_request(new_socket, ip_port):
    # 接收客户端传递过来的消息
    recv_data = new_socket.recv(1024)
    # 判断数据是否为空
    if recv_data:
        recv_data = recv_data.decode('utf-8')
        # 8、获取用户请求的资源信息（IP地址 => 后面资源路径）
        recv_list = recv_data.split(' ', maxsplit=2)
        request_path = recv_list[1]
        # 9、判断路径是否为/斜杠，斜杠代表默认访问首页
        if request_path == '/':
            request_path = '/index.html'

        # 10、根据用户的请求，返回对应的页面给客户端浏览器
        try:
            with open('source/html' + request_path, 'rb') as f:
                data = f.read()
        except:
            # 文件未找到，返回404（http响应报文）
            # ① 响应行 ② 响应头 ③ 空行 ④ 响应体
            response_line = 'HTTP/1.1 404 Not Found\r\n'
            response_header = 'Server:PWB2.0\r\nContent-type:text/html; charset=utf-8\r\n'
            response_body = '很抱歉，您要访问的资源不存在，404 Not Found'

            # 返回数据
            response_data = (response_line + response_header + '\r\n' + response_body).encode('utf-8')
            new_socket.send(response_data)
        else:
            # 文件找到了，返回200（http响应报文）
            response_line = 'HTTP/1.1 200 OK\r\n'
            response_header = 'Server:PWB2.0\r\n'
            response_body = data

            # 返回数据
            response_data = (response_line + response_header + '\r\n').encode('utf-8') + response_body
            new_socket.send(response_data)
        finally:
            # 关闭新产生的套接字，相当一个请求已经处理完毕了
            new_socket.close()

# 2、定义程序执行入口
if __name__ == '__main__':
    # 3、创建套接字对象
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 端口复用
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    # 4、绑定IP和端口号
    tcp_server_socket.bind(("", 9000))
    # 5、开始监听
    tcp_server_socket.listen()
    # 6、等待客户端连接
    while True:
        new_socket, ip_port = tcp_server_socket.accept()
        # 创建子线程
        sub_thread = threading.Thread(target=handle_client_request, args=(new_socket, ip_port))
        # 启动子线程
        sub_thread.start()