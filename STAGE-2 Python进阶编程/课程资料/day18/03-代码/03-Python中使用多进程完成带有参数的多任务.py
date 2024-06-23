# 1、导入进程模块
import multiprocessing
import time

# 定义一个music函数
def music(num):
    for i in range(num):
        print('听音乐')
        time.sleep(0.2)


# 定义一个coding函数
def coding(t):
    for i in range(3):
        print('敲代码')
        time.sleep(t)


# 定义程序的执行入口
if __name__ == '__main__':
    # 2、创建进程对象
    music_process = multiprocessing.Process(target=music, args=(3,))
    coding_process = multiprocessing.Process(target=coding, kwargs={'t':0.2})

    # 3、启动进程执行多任务
    music_process.start()
    coding_process.start()
