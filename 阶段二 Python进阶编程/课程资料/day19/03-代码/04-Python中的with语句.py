'''
with语句属于Python高级语法
等价于
try:
except:
finally:
'''
with open('python.txt', 'w') as f:
    # 文件操作（读写）
    f.write('hello world')

