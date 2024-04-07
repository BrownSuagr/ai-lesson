import multiprocessing
import time
import os


# 定义音乐函数
def music(num):
    print(f'参数num：{num}')
    for i in range(num):
        print(f'听音乐:{i}')
        time.sleep(0.1)


# 定义写代码函数
def coding(t):
    print(f'参数t：{t}')
    for j in range(t):
        print(f'写代码:{j}')
        time.sleep(0.1)


def work():
    print('执行work任务……')

    pid = os.getpid()
    print(f'work进程编号：{pid}')

    ppid = os.getppid()
    print(f'work父进程编号：{ppid}')


def main():
    print('Main function entry')

    # 二、进程的实现及参数传递方式
    # 进程执行带有参数的任务传参的两种方式：
    # 1、元组方式传参：元组方式传参一定要和参数顺序保持一致
    # 2、字典方式传参：字典方式传参中key一定要和参数名称保持一致

    music_process = multiprocessing.Process(target=music, args=(3,))
    coding_process = multiprocessing.Process(target=coding, kwargs={'t': 10})

    music_process.start()
    coding_process.start()

    # 三、获取进程编号
    # 1、获取父进程编号
    parent_pid = os.getpid()
    time.sleep(1)
    print(f'parent_pid:{parent_pid}')

    multiprocessing_parent_pid = multiprocessing.current_process().pid
    time.sleep(1)
    print(f'multiprocessing_parent_pid:{multiprocessing_parent_pid}')

    music_pid = music_process.pid
    time.sleep(1)
    print(f'music_pid:{music_pid}')

    coding_pid = coding_process.pid
    time.sleep(1)
    print(f'coding_pid:{coding_pid}')





if __name__ == '__main__':
    main()
