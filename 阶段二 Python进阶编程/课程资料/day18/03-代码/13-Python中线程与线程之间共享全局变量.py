# 导入模块
import threading
import time

# 定义一个全局变量
my_list = []

# 定义write_data方法
def write_data():
    for i in range(3):
        print('add：', i)
        my_list.append(i)

# 定义read_data方法
def read_data():
    print(my_list)

if __name__ == '__main__':
    # 创建子线程对象
    write_thread = threading.Thread(target=write_data)
    read_thread = threading.Thread(target=read_data)

    # 启动子线程对象
    write_thread.start()
    time.sleep(1)
    read_thread.start()