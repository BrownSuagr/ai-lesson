# 导入模块
import os
import multiprocessing
import time

# 方案一：使用os模块获取当前进程PID
# pid = os.getpid()
# print(pid)

# 方案二：使用multiprocessing获取当前进程PID
pid = multiprocessing.current_process().pid
print(pid)
time.sleep(20)