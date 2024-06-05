# 1、导入模块
import threading
import time

# 定义一个music函数
def music():
    for i in range(10):
        print('听音乐')
        time.sleep(0.2)

# 定义一个coding函数
def coding():
    for i in range(10):
        print('敲代码')
        time.sleep(0.2)

# 定义程序的执行入口
if __name__ == '__main__':
    # 2、创建线程对象
    music_thread = threading.Thread(target=music)
    coding_thread = threading.Thread(target=coding)

    # 3、启动线程
    music_thread.start()
    coding_thread.start()