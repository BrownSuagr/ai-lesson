# 1、打开文件
f = open('python.txt', 'r')
# 2、读写文件
data = f.read()
print(data)
# 3、关闭文件
f.close()