from src.karel_symexecution.utils.enums import Quadrant


def get_position_from_quadrant(quadrant: Quadrant, rows: int, cols: int):
    if quadrant == Quadrant.bottom_left:
        if rows <= 2:
            row = 0
        else:
            row = 1

        if cols <= 2:
            col = 0
        else:
            col = 1

    elif quadrant == Quadrant.bottom_right:
        if rows <= 2:
            row = 0
        else:
            row = 1

        if cols <= 2:
            col = cols - 1
        else:
            col = cols - 2

    elif quadrant == Quadrant.top_left:
        if rows <= 2:
            row = rows - 1
        else:
            row = rows - 2

        if cols <= 2:
            col = 0
        else:
            col = 1

    elif quadrant == Quadrant.top_right:
        if rows <= 2:
            row = rows - 1
        else:
            row = rows - 2

        if cols <= 2:
            col = cols - 1
        else:
            col = cols - 2

    elif quadrant == Quadrant.center:
        row = rows // 2
        col = cols // 2

    else:
        raise ValueError(f'Unknown quadrant {quadrant}')

    return row, col


def get_quadrant_from_position(row: int, col: int, rows: int, cols: int):
    if rows <= 2 and cols <= 2:
        if row == 0 and col == 0:
            return Quadrant.bottom_left
        elif row == 0 and col == cols - 1:
            return Quadrant.bottom_right
        elif row == rows - 1 and col == 0:
            return Quadrant.top_left
        elif row == rows - 1 and col == cols - 1:
            return Quadrant.top_right

    elif rows <= 2:
        if cols // 3 <= col < 2 * cols // 3:
            quadrant = Quadrant.center
        elif col < cols // 3:
            if row == 0:
                quadrant = Quadrant.bottom_left
            elif row == 1:
                quadrant = Quadrant.top_left
        elif col >= 2 * cols // 3:
            if row == 0:
                quadrant = Quadrant.bottom_right
            elif row == 1:
                quadrant = Quadrant.top_right

    elif cols <= 2:
        if rows // 3 <= row < 2 * rows // 3:
            quadrant = Quadrant.center
        elif row < rows // 3:
            if col == 0:
                quadrant = Quadrant.bottom_left
            elif col == 1:
                quadrant = Quadrant.top_left
        elif row >= 2 * rows // 3:
            if col == 0:
                quadrant = Quadrant.bottom_right
            elif col == 1:
                quadrant = Quadrant.top_right

    elif rows // 3 <= row < 2 * rows // 3 and cols // 3 <= col < 2 * cols // 3:
        quadrant = Quadrant.center

    elif row < rows // 2 and col < cols // 2:
        quadrant = Quadrant.bottom_left

    elif row < rows // 2 and col >= cols // 2:
        quadrant = Quadrant.bottom_right

    elif row >= rows // 2 and col < cols // 2:
        quadrant = Quadrant.top_left

    elif row >= rows // 2 and col >= cols // 2:
        quadrant = Quadrant.top_right

    else:
        raise ValueError(f'Unknown position {row}, {col}')

    return quadrant
