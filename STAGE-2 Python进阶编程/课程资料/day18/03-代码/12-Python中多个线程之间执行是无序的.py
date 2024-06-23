# 1、导入模块
import threading
import time

# 定义一个work任务
def work():
    # 休眠0.5s
    time.sleep(0.5)
    current_thread = threading.current_thread()  # 用于获取当前线程的信息
    print(current_thread)

# 定义程序的执行入口
if __name__ == '__main__':
    for i in range(10):
        sub_thread = threading.Thread(target=work)
        sub_thread.start()