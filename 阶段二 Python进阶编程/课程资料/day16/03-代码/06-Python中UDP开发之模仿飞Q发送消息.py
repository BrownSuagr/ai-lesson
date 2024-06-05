# 导入模块
import socket

# 1、创建套接字对象
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2、使用udp套接字对象发送信息
for i in range(10):
    content = '1:134871264:蔡徐坤:cndws-pc:32:你好，陌生人！'.encode('gbk')
    udp_socket.sendto(content, ("192.168.27.93", 2425))

udp_socket.close()

# udp广播 => 192.168.27.xxx => 192.168.27.255（广播地址）