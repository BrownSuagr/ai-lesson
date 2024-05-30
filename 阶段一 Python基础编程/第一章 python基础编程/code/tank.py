from time import sleep
from pygame import display
from pygame import Color

from sys import exit
from pygame import event
from pygame import QUIT
from pygame import quit

from pygame.sprite import Sprite
from pygame.sprite import AbstractGroup
from pygame import image


SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500
BG_COLOR = Color(0, 0, 0)

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPEED = 1


class BaseItem(Sprite):
    pass


class Tank(BaseItem):
    pass


class OurTank(Tank):
    tanks = []
    count = 0

    __obj = None
    __init_flag = True

    def __new__(cls, *args, **kwargs):
        if cls.__obj is None:
            cls.__obj = object.__new__(cls)
        return cls.__obj

    def __init__(self, left, top):
        super().__init__()
        if OurTank.__init_flag:
            OurTank.__init_flag = False


class BaseItem(Sprite):
    def __init__(self, *groups: AbstractGroup):
        super().__init__(*groups)


class Tank(BaseItem):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SPEED = 1

    def __init__(self, *groups: AbstractGroup):
        super().__init__(*groups)
        self.images = {}
        self.direction = Tank.UP
        self.my_image = None
        self.rect = None
        self.move_switch = None
        self.bullets = []

    def displayTank(self):
        self.my_image = self.images.get(self.direction, None)
        MainGame.window.blit(source=self.my_image, dest=self.rect)


class MainGame:
    window = None

    def __init__(self):
        pass

    def start_game(self):
        display.init()
        MainGame.window = display.set_mode(size=[SCREEN_WIDTH, SCREEN_HEIGHT])
        display.set_caption("Tank War 1.0")
        while True:
            sleep(0.002)
            MainGame.window.fill(color=BG_COLOR)
            self.getEvent()
            display.update()

    @staticmethod
    def gameOver():
        quit()
        exit()

    def getEvent(self):
        for e in event.get():
            if e.type == QUIT:
                self.gameOver()


if __name__ == "__main__":
    my_game = MainGame()
    my_game.start_game()










