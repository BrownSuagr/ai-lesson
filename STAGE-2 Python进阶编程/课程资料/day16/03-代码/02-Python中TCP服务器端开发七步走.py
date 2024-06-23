# 导入模块
import socket

# 1、创建TCP服务器端套接字对象
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(tcp_server_socket)

# 2、绑定IP地址与端口号（本机或当前服务器）,注意参数的数据类型要求是一个元组
tcp_server_socket.bind(("", 8000))

# 3、设置监听（让代码监听端口传输过来的数据）,代表允许最大的连接数
tcp_server_socket.listen(128)

# 4、等待客户端连接（难点）
# print(tcp_server_socket.accept())
# 元组拆包，tcp_server_socket.accept()结果是一个元组，有两个元素，第一个元素是一个新的套接字对象，第二个元素是客户端的信息
# 客户端与服务器端连接成功以后，信息的发送与接收都要依靠新产生的套接字对象，因为其内部保留客户端与服务器端的相关信息
new_socket, ip_port = tcp_server_socket.accept()
# print(ip_port)

# 5、接收客户端发送过来的数据
recv_data = new_socket.recv(1024) # 返回结果=>字节流
if recv_data:
    # 把接收到的字节流数据转换为字符串格式
    recv_data = recv_data.decode('gbk')
    print(f'{ip_port}客户端发送过来的数据：{recv_data}')

    # 6、处理客户端请求并返回数据给客户端
    content = '信息已收到，over，over！'.encode('gbk')
    new_socket.send(content)

# 7、当数据全部处理完毕后，关闭套接字对象（关闭新产生的套接字对象与服务器端套接字对象）
new_socket.close()
tcp_server_socket.close()