import requests

data = requests.get('http://127.0.0.1:8000/images/0.jpg')
# 图片不需要解码
getpic = data.content  # 字节流数据

# 使用with语句创建一个本地文件
with open('./itcast.jpg', 'wb') as f:
    f.write(getpic)