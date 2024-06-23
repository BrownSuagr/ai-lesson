# 导入模块
import multiprocessing
import time

# 创建一个任务work
def work():
    for i in range(10):
        print('work子进程正在执行...')
        time.sleep(0.2)

# 定义程序执行入口
if __name__ == '__main__':
    # 创建子进程对象
    work_process = multiprocessing.Process(target=work)
    # 方案一：设置守护主进程
    # work_process.daemon = True
    # 启动子进程
    work_process.start()
    # 主进程休眠1s
    time.sleep(1)
    # 方案二：在主进程结束之前，强制销毁所有子进程
    # work_process.terminate()
    print('主进程执行结束！')