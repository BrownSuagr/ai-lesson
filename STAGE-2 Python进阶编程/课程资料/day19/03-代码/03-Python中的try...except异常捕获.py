# 定义异常处理语句
try:
    f = open('python.txt', 'r')
    f.write('hello world')
except:
    print('打开模式不正确，文件内容无法正常写入')
finally:
    f.close()