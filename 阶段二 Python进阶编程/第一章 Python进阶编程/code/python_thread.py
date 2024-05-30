# 导入多进程的包
import multiprocessing
import os
# 导入线程模块
import threading
import time


# 音乐函数
def music():
    print('执行music函数')
    for i in range(5):
        print(f'音乐线程执行{i}')
        time.sleep(0.2)


# 音乐函数
def music(num, content):
    print(f'执行music函数, 参数:{content}')
    for i in range(num):
        print(f'音乐线程执行{i}')
        time.sleep(0.2)


# 写代码函数
def coding():
    print('执行coding函数')
    for j in range(4):
        print(f'写代码线程{j}')
        time.sleep(0.2)


# 写代码函数
def coding(num, content):
    print(f'执行coding函数 参数:{content}')
    for j in range(num):
        print(f'写代码线程{j}')
        time.sleep(0.2)


# 主函数入口
def main():
    print('主程序入口')

    print(f'主线程PID:{os.getgid()}')

    music_thread = threading.Thread(target=music)
    coding_thread = threading.Thread(target=coding)
    music_thread.start()
    coding_thread.start()

    music_thread_param = threading.Thread(target=music, args=(10, '听音乐'))
    coding_thread_param = threading.Thread(target=coding, kwargs={'num': 20, 'content': '敲代码'})
    music_thread_param.start()
    coding_thread_param.start()

    # 多线程之间执行是无序的

    # 进程和线程的对比
    # 1、线程不能独立执行，只能在进程基础上
    # 2、进程之间不能共享全局变量，线程之间可以共享全局变量
    # 3、创建进程的资源开销要比创建线程资源开销大
    # 4、进程是操作系统资源分配的最小单位，线程是CPU调度的基本单位




if __name__ == '__main__':
    main()
