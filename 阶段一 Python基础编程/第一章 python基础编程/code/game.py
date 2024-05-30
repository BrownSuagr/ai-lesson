import pygame as pg
import time


def main():
    # 创建一个窗口
    mode = pg.display.set_mode((2048, 1024), 0, 32)
    bg = pg.image.load('../temp/bg.jpg')
    mode.blit(bg, (0, 0))
    pg.display.update()
    time.sleep(5)


if __name__ == '__main__':
    main()
