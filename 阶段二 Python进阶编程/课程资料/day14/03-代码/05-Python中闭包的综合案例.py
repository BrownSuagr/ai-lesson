'''
函数的执行流程：
① 首先代码都是从上往下一行一行执行的，所以首先在内存中定义函数func，其内部代码并没有真正执行，继续向下，执行f = func()
② 执行func然后把结果赋值给变量f，当执行f(1)时，首先执行result=0进行赋值操作，然后执行inner函数，nolocal代表使用上一级
的result变量，result + num则结果为1打印输出。到这里，代码执行完毕后，result变量没有被回收，此时result=1
③ 继续向下执行，执行f(2)，把2赋值给num参数，然后继续向下执行，result += num，由于上一步，result变量并没有被内存回收，
所以result的值还是上一次的执行结果，本次执行，在上一级的基础上进行+2操作，所以结果为3，到这里，result依然没有被回收
④ f(3)，相当于把3赋值给参数num，所以result += num，相当于使用上一次result的结果 + 3，所以结果为6
'''
def func():
    result = 0
    def inner(num):
        nonlocal result
        result += num
        print(result)
    return inner

f = func()  # 把func返回结果赋值给f，func函数返回inner函数内存地址，f = inner = inner所指向的内存地址
f(1)  # 1
f(2)  # 3
f(3)  # 6