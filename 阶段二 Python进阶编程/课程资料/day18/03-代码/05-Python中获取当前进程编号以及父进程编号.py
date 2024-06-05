# 导入模块
import os
import multiprocessing

# 定义一个函数work
def work():
    print('执行work任务...')
    # 获取当前进程编号pid
    pid = os.getpid()
    print(f'work子进程编号：{pid}')
    # 获取当前父进程编号ppid
    ppid = os.getppid()
    print(f'当前进程的父进程编号：{ppid}')

# 定义程序的执行入口
if __name__ == '__main__':
    print(f'主进程编号：{os.getpid()}')
    sub_process = multiprocessing.Process(target=work)
    sub_process.start()