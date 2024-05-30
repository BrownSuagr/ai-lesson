import turtle


# 事件处理器的工厂函数
def get_key_event_handler(t):
    # 内嵌函数作为事件处理器
    def key_event_handler(event):
        print('监听函数')
        if event.keysym == "Up":
            t.forward(10)
        elif event.keysym == "Down":
            t.backward(10)
        elif event.keysym == "Left":
            t.left(10)
        elif event.keysym == "Right":
            t.right(10)
    return key_event_handler


def main():
    # 创建画布和turtle
    screen = turtle.Screen()
    t = turtle.Turtle()

    # 为事件处理器传递turtle对象
    key_event_handler = get_key_event_handler(t)

    # 注册事件处理器
    screen.onkeypress(key_event_handler, "Up")
    screen.onkeypress(key_event_handler, "Down")
    screen.onkeypress(key_event_handler, "Left")
    screen.onkeypress(key_event_handler, "Right")

    # 启动事件循环
    screen.listen()
    screen.mainloop()


if __name__ == '__main__':
    main()

