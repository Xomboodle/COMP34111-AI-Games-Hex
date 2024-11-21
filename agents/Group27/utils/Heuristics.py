"""TODO"""

from src.Board import Board
from src.Colour import Colour
import copy
import math

class Heuristics:
    @staticmethod
    def evaluateBoard(boardState: Board, player: bool, hyps: list[float]) -> float:
        """Evaluate the board state"""
        self_shortest_path = 11
        opponent_shortest_path = 11
        tiles_in_self_shortest = 0
        tiles_in_opponent_shortest = 0

        boardCopy = copy.deepcopy(boardState)
        if boardCopy.has_ended(Colour.BLUE if player else Colour.RED):
            return 100
        elif boardCopy.has_ended(Colour.RED if player else Colour.BLUE):
            return -math.inf

        # Scuffy, but faster
        longest_red = 11
        longest_blue = 11
        red_tiles = 0
        blue_tiles = 0
        for i in range(boardCopy.size):
            for j in range(boardCopy.size):
                tile_colour = boardCopy.tiles[i][j].colour
                pre_row_colour = boardCopy.tiles[i-1][j].colour if i > 0 else 2
                pre_col_colour = boardCopy.tiles[i][j-1].colour if j > 0 else 2
                if tile_colour == Colour.RED:
                    red_tiles += 1
                    if pre_row_colour in [Colour.RED, 2]:
                        longest_blue += 1
                elif tile_colour == Colour.BLUE:
                    blue_tiles += 1
                    if pre_col_colour in [Colour.BLUE, 2]:
                        longest_red += 1

        average_red_path = int((11 + longest_red) / 2)
        average_blue_path = int((11 + longest_blue) / 2)

        # It would be reasonable to suggest that some of the existing tiles are in the shortest path (25%)
        if player:
            self_shortest_path = average_blue_path
            opponent_shortest_path = average_red_path
            tiles_in_self_shortest = blue_tiles // 4
            tiles_in_opponent_shortest = longest_red // 2
        else:
            self_shortest_path = average_red_path
            opponent_shortest_path = average_blue_path
            tiles_in_self_shortest = red_tiles // 4
            tiles_in_opponent_shortest = longest_blue // 2

        heuristic = (hyps[0] * (1 / self_shortest_path)) - (hyps[1] * (1 /opponent_shortest_path)) + (hyps[2] * tiles_in_self_shortest) - (hyps[3] * tiles_in_opponent_shortest)

        if player and longest_blue >= 22:
            return -math.inf
        if player and longest_red >= 22:
            return math.inf
        if not player and longest_red >= 22:
            return -math.inf
        if not player and longest_blue >= 22:
            return math.inf
        return heuristic
    
    def evaluate_basic(board : Board, player : bool):
        red_result = board.has_ended(Colour.BLUE)
        blue_result = board.has_ended(Colour.RED)
        if player:
            # blue
            if blue_result:
                return 1
            if red_result:
                return -1
            return 0
        else:
            #red
            if red_result:
                return 1
            if blue_result:
                return -1
            return 0