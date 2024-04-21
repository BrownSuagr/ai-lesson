
# 定义日志装饰器
def log(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"执行了有参数有返回值且带参数的装饰器，装饰器参数：{level} 函数名：{func.__name__} args：{args} kwargs:{kwargs}")
            if 'INFO' == level:
                print(f'执行日志级别为：{level}')
            elif 'WARN' == level:
                print(f'执行日志级别为：{level}')
            else:
                print(f'执行日志级别为：{level}')
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log2():
    def decorator(func):
        def wrapper():
            print(f"执行了无参数和无返回值的装饰器，函数名： {func.__name__}")
            func()
        return wrapper
    return decorator

def log3():
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"执行了有参数和有返回值的装饰器，函数名：{func.__name__} args：{args} kwargs:{kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 目标函数（指定参数名）
@log(level="INFO")
def targetAdd(param1, param2):
    print(f'执行了目标函数 参数一：{param1} 参数二：{param2}')
    return param1 + param2

# 目标函数（指定参数名）
@log(level="WARN")
def targetSub(param1, param2):
    print(f'执行了目标函数 参数一：{param1} 参数二：{param2}')
    return param1 - param2

@log2()
def comment():
    print('执行了评论函数')

@log3()
def target(param1, param2):
    print(f'执行了目标函数 参数一：{param1} 参数二：{param2}')
    return param1 + param2




# 主函数入口
def main():
    # print(add(1, 2))
    # comment()
    # result = target(1, 2)
    # print(f'返回结果：{result}')

    result1 = targetAdd(2, 2)
    print(f'返回结果：{result1}')

    result2 = targetSub(3, 1)
    print(f'返回结果：{result2}')



if __name__ == '__main__':
    main()



