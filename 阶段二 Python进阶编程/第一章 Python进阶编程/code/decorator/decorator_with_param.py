
# 定义日志装饰器
def log(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"参数：[{level}] 函数名： {func.__name__} args： {args} kwargs： {kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 目标函数
@log(level="INFO")
def add(a, b):
    return a + b


# 主函数入口
def main():
    print(add(1, 2))


if __name__ == '__main__':
    main()



