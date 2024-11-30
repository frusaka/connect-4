from collections import OrderedDict
import numpy as np
import math
import time


class LRUCache:
    def __init__(self, size=1_000_000):
        self.size = size
        self.data = OrderedDict()

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        if len(self.data) >= self.size:
            self.data.popitem(False)
        self.data[key] = value

    def clear(self):
        self.data.clear()


class Board:
    terminal_weight = 10_000

    def __init__(self):
        self.data = np.zeros((6, 7), dtype=np.int8)
        self.turn = -1
        self.prev = None
        self._state = None
        self.evals = dict(
            [
                (1, np.zeros((6, 7), dtype=np.int32)),
                (-1, np.zeros((6, 7), dtype=np.int32)),
            ]
        )

    @property
    def game_state(self):
        return self._state

    def update_state(self):
        if winner := self.check_win():
            self._state = int(winner)
        elif np.all(self.data[0]):
            self._state = 0
        else:
            self._state = None

    def windows(self):
        # Check horizontal
        for row in range(6):
            yield self.data[row]

        # Check vertical wins
        for col in range(7):
            yield self.data[:, col]

        # Check diagonal (top-left to bottom-right)
        yield np.diagonal(self.data)
        for col in range(1, 4):
            yield np.diagonal(self.data, offset=col)
        for col in range(1, 3):
            yield np.diagonal(self.data, offset=-col)

        # Check diagonal (bottom-left to top-right)
        yield np.diagonal(np.fliplr(self.data))
        for col in range(1, 4):
            yield np.diagonal(np.fliplr(self.data), offset=col)
        for col in range(1, 3):
            yield np.diagonal(np.fliplr(self.data), offset=-col)

    def compress(self):
        mask = 0
        data = self.data.flatten()
        for i in range(42):
            cell = data[i]
            if cell == -1:
                cell = 2
            mask |= int(cell) << (i * 2)
        return mask

    def legal_moves(self):
        for col in range(7):
            for row in reversed(range(6)):
                if self.data[row, col]:
                    continue
                yield row, col
                break

    def sorted_moves(self):
        return sorted(
            self.legal_moves(),
            key=lambda move: self.evals[self.turn][move],
            reverse=self.turn == 1,
        )

    def make_move(self, move):
        self.data[move] = self.turn
        if self.evals[self.turn][move]:
            self.evals[self.turn][move] = self.eval(0.1)
        self.prev = move
        self.turn = -self.turn
        self.update_state()

    def drop_piece(self, col):
        for row in reversed(range(6)):
            if self.data[row, col] == 0:
                self.make_move((row, col))
                return row
        return -1

    def undo_move(self, move):
        self.data[move] = 0
        self.turn = -self.turn
        self.update_state()

    def updateEval(self, player, move, value):
        self.evals[player][move] = value

    def move_heuristic(self, player, move=None):
        def consecutive_sum(window):
            l, res = 0, 0

            for r in range(len(window)):
                if r == len(window) - 1 or (window[r] != player and window[r]):
                    if (r - l) >= 4:
                        val = abs(np.sum(window[l:r]))
                        potential = 0
                        if val < (r - l):
                            potential = 2 ** min((r - l) - val - 1, 4) * 0.5
                        if val > 0:
                            val = 2 ** min(val - 1, 3)
                        res = max(res, val + potential)

                    l = r

            return res * player

        eval = 0
        prev = self.data[move]

        if move:
            self.data[move] = player
        for window in self.windows():
            eval += consecutive_sum(window)
        if move:
            self.data[move] = prev

        return eval

    def center_control(self):
        weight = 0
        for row in range(6):
            for col in range(7):
                if not self.data[row, col]:
                    continue
                weight += min(0, 3 - abs(row - 3)) * self.data[row, col]

        return weight

    def best_move(self, timeout=2_000):
        start = time.time()
        for depth in range(1, 100, 2):
            if (time.time() - start) * 1000 >= timeout:
                depth -= 2
                break
            move, score = self.minimax(
                0, depth, self.turn == 1, -math.inf, math.inf, {}
            )
            if score == abs(self.terminal_weight) or abs(score) >= 984:
                break
        time.sleep(max(0, (timeout / 1000) - (time.time() - start)))
        print("Computer Eval:", round(score, 2), "Depth:", depth)
        return move, score

    def minimax(self, depth, max_depth, is_maximizing, alpha, beta, seen):
        state = is_maximizing, self.compress()

        if state in seen:
            return seen[state][1]

        if depth == max_depth or self.game_state is not None:
            score = self.eval(depth if depth == max_depth else 1000)
            val = (None, score)

            if score == self.terminal_weight:
                seen[state] = (depth,)
                val = (None, 1000 - depth)

            elif score == -self.terminal_weight:
                val = (None, depth - 1000)
            seen[state] = (depth, val)
            return val

        if is_maximizing:
            best_score = -math.inf
            for move in self.sorted_moves():
                self.make_move(move)
                _, score = self.minimax(depth + 1, max_depth, False, alpha, beta, seen)
                self.undo_move(move)

                self.updateEval(1, move, score)

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)

                if score == self.terminal_weight or alpha >= beta:
                    break
        else:
            best_score = math.inf
            for move in self.sorted_moves():
                self.make_move(move)
                _, score = self.minimax(depth + 1, max_depth, True, alpha, beta, seen)
                self.undo_move(move)

                self.updateEval(-1, move, score)

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)

                if not score or score == -self.terminal_weight or beta <= alpha:
                    break
        seen[state] = (depth, (best_move, best_score))
        return (best_move, best_score)

    def eval(self, depth):
        if self.game_state is not None:
            return self.game_state * self.terminal_weight
        prev, move = -self.turn, self.prev
        r = self.evals[prev][move]
        h = self.move_heuristic(1) + self.move_heuristic(-1) + self.center_control()
        kmin = 0.5
        kmax = 4
        w2 = depth / (depth + (kmax / depth + kmin))
        w1 = 1 - w2
        return w1 * h + w2 * r

    def check_win(self):
        for window in self.windows():
            for offset in range(len(window) - 3):
                if not window[offset]:
                    continue
                if abs(np.sum(window[offset : offset + 4])) == 4:
                    self._state = window[offset]
                    return window[offset]
        return 0

    def reset(self):
        self.data = np.zeros((6, 7), dtype=np.int8)
        self._state = None
        self.turn = -1
        return self.data.tolist()
