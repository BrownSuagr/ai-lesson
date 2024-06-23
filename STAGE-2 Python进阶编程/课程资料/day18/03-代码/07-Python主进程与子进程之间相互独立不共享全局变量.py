# 导入模块
import multiprocessing
import time

# 创建一个全局变量（主进程中的全局变量）
my_list = []

# 创建一个任务，用于向my_list中添加数据
def write_data():
    print(id(my_list))
    for i in range(3):
        my_list.append(i)
        print('add:', i)
    print(f'write_data：{my_list}')

# 创建一个任务，用于从my_list中获取数据
def read_data():
    print(id(my_list))
    print(f'read_data: {my_list}')

# 定义程序执行入口
if __name__ == '__main__':
    # 输出my_list全局变量内存地址
    print(id(my_list))

    # 创建子进程
    write_process = multiprocessing.Process(target=write_data)
    read_process = multiprocessing.Process(target=read_data)

    # 启动子进程
    write_process.start()
    time.sleep(1)
    read_process.start()