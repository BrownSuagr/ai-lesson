# 导入模块
from fastapi import FastAPI
from fastapi import Response

import uvicorn

# 创建FastAPI对象
app = FastAPI()

# 使用路由装饰器进行图片信息收发
@app.get('/images/0.jpg')
def func_0():
    with open('source/images/0.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/1.jpg')
def func_0():
    with open('source/images/1.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/2.jpg')
def func_0():
    with open('source/images/2.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/3.jpg')
def func_0():
    with open('source/images/3.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/4.jpg')
def func_0():
    with open('source/images/4.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/5.jpg')
def func_0():
    with open('source/images/5.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行图片信息收发
@app.get('/images/6.jpg')
def func_0():
    with open('source/images/6.jpg', 'rb') as f:
        data = f.read()
    return Response(content=data, media_type='jpg')

# 使用路由装饰器进行信息收发
@app.get('/index.html')
def main():
    with open('source/html/index.html', 'rb') as f:
        data = f.read()

    return Response(content=data, media_type='text/html')

# 使用uvicorn启动服务（相当于while True）
uvicorn.run(app, host='127.0.0.1', port=8000)