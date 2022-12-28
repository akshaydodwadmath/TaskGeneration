from enum import Enum


class Direction(str, Enum):
    north = 'north'
    east = 'east'
    south = 'south'
    west = 'west'
    any = 'any'

    def __str__(self):
        return self.value


class Quadrant(str, Enum):
    top_left = 'top_left'
    top_right = 'top_right'
    bottom_left = 'bottom_left'
    bottom_right = 'bottom_right'
    center = 'center'

    def __str__(self):
        return self.value
