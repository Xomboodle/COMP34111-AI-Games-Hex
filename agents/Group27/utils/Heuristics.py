"""TODO"""

from src.Board import Board
from src.Colour import Colour
import copy
import math
from collections import deque

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
            #red
            if red_result:
                return 1
            if blue_result:
                return -1
            return 0
        else:    
            # blue
            if blue_result:
                return 1
            if red_result:
                return -1
            return 0
        
    @staticmethod
    def evaluateBoard2(boardState: Board, player: bool) -> float:
        """
            Evaluates the board state by finding the shortest path via
            BFS, and how many tiles within it are currently occupied by
            the player
        """
        path_length, path = bfs(boardState, player)
        path_length_opp, path_opp = bfs(boardState, not(player))

        colour = Colour.BLUE if player else Colour.RED
        colour_opp = Colour.RED if player else Colour.BLUE
        placed = 0
        placed_opp = 0
        for tile in path:
            if boardState.tiles[tile[0]][tile[1]].colour == colour:
                placed += 1

        for tile in path_opp:
            if boardState.tiles[tile[0]][tile[1]].colour == colour_opp:
                placed_opp += 1

        # TODO:
        #  Values are hard-coded for now, will need to change
        heuristic = (0.6 * path_length) + (0.8 * placed) - (0.3 * path_length_opp) - (0.7 * placed_opp)
        return heuristic
    
def bfs(board: Board, player: bool):
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    if player: # BLUE PLAYER
        opp_colour = Colour.RED
        start_nodes = [(0, j) for j in range(11) if board.tiles[0][j].colour != Colour.RED]
        target_side = lambda x, y: x == 10
    else: # RED PLAYER
        opp_colour = Colour.BLUE
        start_nodes = [(i, 0) for i in range(11) if board.tiles[i][0].colour != Colour.BLUE]
        target_side = lambda x, y: y == 10
    
    queue = deque([(x, y, [(x, y)]) for x, y in start_nodes])  # (x, y, path_so_far)
    visited = set(start_nodes)
    while queue:
        x, y, path = queue.popleft()

        # Check if we've reached the target side
        if target_side(x, y):
            return len(path), path

        # Explore neighbours
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and if the neighbour doesn't belong to the opponent
            if 0 <= nx < 11 and 0 <= ny < 11 and board.tiles[nx][ny].colour != opp_colour and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [(nx, ny)]))


    return -1, [];
