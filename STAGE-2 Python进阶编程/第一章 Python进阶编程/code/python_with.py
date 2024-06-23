import memory_profiler as mem


def main():
    print('')

    # 1、with语句
    #   基本概念：
    #       1、生成器不保存具体的数据，而是保存一个结果对象，对象中存储了所有数据的生成规则
    #       2、生成器中又一个next()函数，每次调用next()就会生成对应元素
    #       3、因生成器中只存储规则，不存储数据。可以降低对资源的利用
    #   生成方式：
    #       1、推倒式生成器
    #       2、yield生成器
    # 2、yield生成器
    #   基本概念：
    #       1、首先定义一个函数，生成器是一个对象而不是一个函数
    #       2、函数内部存在一个关键字yield
    # 3、深浅拷贝
    # 4、正则表达式

    start = mem.memory_usage()
    print(f'start:{start}')
    square_nums = [i * i for i in range(100000)]
    # square_nums = (i * i for i in range(100000))
    end = mem.memory_usage()
    print(f'end:{end}')

    print(i for i in range(3))


if __name__ == '__main__':
    main()
