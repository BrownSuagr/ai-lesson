# 1、导入模块
import threading
import time

# 定义一个work函数
def work():
    for i in range(10):
        print('work子线程正在执行...')
        time.sleep(0.2)

# 定义程序执行入口
if __name__ == '__main__':
    # 产生一个主线程（默认有程序运行就会自动产生）
    # 2、创建子线程对象
    # 方案一：把子线程设置为守护主线程
    # sub_thread = threading.Thread(target=work, daemon=True)
    sub_thread = threading.Thread(target=work)
    # 方案二：使用setDeamon(True)方法设置守护主线程
    sub_thread.setDaemon(True)
    # 3、启动子线程
    sub_thread.start()
    # 让主线程休眠1s
    time.sleep(1)
    # 输出信息
    print('主线程执行结束！')
