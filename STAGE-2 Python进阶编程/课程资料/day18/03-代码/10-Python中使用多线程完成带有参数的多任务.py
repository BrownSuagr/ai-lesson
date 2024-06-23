# 1、导入模块
import threading
import time

# 定义一个music函数
def music(num, content):
    for i in range(num):
        print(content)
        time.sleep(0.2)

# 定义一个coding函数
def coding(num, content):
    for i in range(num):
        print(content)
        time.sleep(0.2)

# 定义程序的执行入口
if __name__ == '__main__':
    # 2、创建线程对象
    music_thread = threading.Thread(target=music, args=(10, '听音乐'))
    coding_thread = threading.Thread(target=coding, kwargs={'num':10, 'content':'敲代码'})

    # 3、启动线程
    music_thread.start()
    coding_thread.start()