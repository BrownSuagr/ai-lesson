# 1、导入requests模块
import requests

# 2、使用requests.get()方法爬取数据，有一个参数url，用于爬取指定url的页面
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36'}
data = requests.get('https://movie.douban.com/', headers=headers)

print(data)
# 3、使用data.content获取爬虫爬取的内容，然后对其进行解码操作（默认返回字节流数据）
print(data.content.decode('utf-8'))
