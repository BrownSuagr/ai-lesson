
def yield_generator(num):
    for i in range(num):
        print('开始生成数据')
        yield i
        print('当前生成结束')

# 生成费波纳契数列
def fibonacci_sequence_generator(num):
    result = [0, 1]
    if 0 == num:
        return result[0:1]
    if 1 == num:
        return result

    for i in range(2, num):
        pre = result[i - 1]
        bef = result[i - 2]
        next_num = pre + bef
        result.append(next_num)
    return result


# yield生成器生成费波纳契数列
def fibonacci_seq_gen_with_yield(num):
    result = [0, 1]

    for i in range(0, num):
        if 0 == i:
            temp = result[0:1]
            yield temp
        elif 1 == i:
            yield result
        else:
            pre = result[i - 1]
            bef = result[i - 2]
            next_num = pre + bef
            result.append(next_num)
            yield result


def main():
    yield_gen = yield_generator(5)
    print(f'第1次生成：{next(yield_gen)}')
    print(f'第2次生成：{next(yield_gen)}')
    print(f'第3次生成：{next(yield_gen)}')
    print(f'第4次生成：{next(yield_gen)}')
    print(f'第5次生成：{next(yield_gen)}')

    for i in yield_gen:
        print(f'第{i}次生成：{i}')

    print(fibonacci_sequence_generator(10))

    fyg = fibonacci_seq_gen_with_yield(10)
    print(f'第1次生成：{next(fyg)}')
    print(f'第2次生成：{next(fyg)}')
    print(f'第3次生成：{next(fyg)}')
    print(f'第4次生成：{next(fyg)}')
    print(f'第5次生成：{next(fyg)}')
    print(f'第6次生成：{next(fyg)}')
    print(f'第7次生成：{next(fyg)}')
    print(f'第8次生成：{next(fyg)}')
    print(f'第9次生成：{next(fyg)}')
    print(f'第10次生成：{next(fyg)}')


if __name__ == '__main__':
    main()