'''
Web静态服务器最少要完成两个工作：① 接收用户请求的资源路径 ② 把用户请求的资源通过Response返回给浏览器端
'''
# 导入模块
from fastapi import FastAPI
from fastapi import Response

import uvicorn

# 创建FastAPI对象
app = FastAPI()

# 使用路由装饰器进行图片信息收发，思路：想个办法，获取用户请求图片地址
# {path}用于获取图片的名称，如0.jpg、1.jpg
# http://127.0.0.1:8000/images/0.jpg => /images/0.jpg路径
@app.get('/images/{path}')
def get_pic(path: str):
    with open(f'source/images/{path}', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')


# 使用路由装饰器进行信息收发
@app.get('/')
def main():
    with open('source/html/index.html', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='text/html')

# 使用路由装饰器进行信息收发
@app.get('/{path}')  # /gdp.html
def get_html(path: str):
    with open(f'source/html/{path}', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='text/html')


# 使用uvicorn启动服务（相当于while True）
uvicorn.run(app, host='127.0.0.1', port=8000)