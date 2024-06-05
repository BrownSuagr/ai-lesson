# 导入socket模块
import socket

# 1、创建套接字对象（socket.AF_INET => IPv4，socket.SOCK_STREAM => TCP协议）
tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2、与服务器端建立连接
tcp_client_socket.connect(("192.168.27.93", 8000))

# 3、发送消息（不能是文本类型，必须是字节流数据）
tcp_client_socket.send('hello, itheima!'.encode(encoding='gbk'))

# 4、接收消息（服务器端返回的数据，默认类型是字节流类型，必须进行解码）
recv_data = tcp_client_socket.recv(1024).decode(encoding='gbk')
print(recv_data)

# 5、关闭套接字对象
tcp_client_socket.close()