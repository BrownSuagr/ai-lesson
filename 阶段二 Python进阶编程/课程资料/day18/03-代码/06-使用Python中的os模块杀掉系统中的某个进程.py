# 导入os模块
import os

# 杀掉某个进程
# pid，要杀死的进程pid
# signal，和操作系统有关，9代表强制杀死进程，15代表正常杀死进程
# kill -9 pid
# kill -15 pid
os.kill(1788, 9)

# 1788进程已被杀死
print('1788进程已被杀死')
