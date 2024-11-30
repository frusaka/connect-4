import eel
import numpy as np
from board import Board

board = Board()
HUMAN = -1
COMPUTER = 1


@eel.expose("dropPiece")
def drop_piece(col):
    return board.drop_piece(col)


@eel.expose("makeMove")
def make_move(move):
    return board.make_move(move)


@eel.expose("gameState")
def game_sate():
    return board.game_state


@eel.expose("bestMove")
def best_move(timeout=1000):
    move, _ = board.best_move(timeout)
    board.make_move(move)
    return move


@eel.expose("getBoard")
def get_board():
    return board.data.tolist()


@eel.expose("getTurn")
def get_turn():
    return board.turn


@eel.expose("resetBoard")
def reset_board():
    return board.reset()


# fmt:off
# board.data = np.array(
# [[ 0,  0,  0,  0,  0,  0,  0],
#  [ 0,  0,  1,  1,  1,  -1,  0],
#  [ 0,  1, -1, -1,  1,  1,  0],
#  [ 0, -1, -1,  1, -1, -1,  1],
#  [ 1, -1,  1, -1,  1,  1,  1],
#  [ 1,  1, 1,  1,  -1, -1,  1]])

# board.data = np.array(
# [[ 0,  -1,   0,   0,   0,   0,   0],
#  [ 0,   0,  -1,   0,   0,   0,   0],
#  [ 0,   0,  -1,  -1,   0,   0,   0],
#  [ 0,   0,   1,   1,  -1,  -1,   0],
#  [ 0,   1,   0,   1,   0,   0,   0],
#  [ 1,  -1,  -1,  -1,   1,   1,   0]])
# fmt:on

# print(board.data)
# print(board.check_win())

eel.init("web")
eel.start("index.html")
