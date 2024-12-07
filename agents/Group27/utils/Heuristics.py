"""TODO"""

from src.Board import Board
from src.Colour import Colour
from agents.Group27.utils.BoardState import BoardState
import copy
import math
from random import randint, choice
from collections import deque
import heapq

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

    @staticmethod
    def evaluate_basic(board : Board, player : bool):
        board.has_ended(Colour.RED)
        board.has_ended(Colour.BLUE)
        winner = board.get_winner()
        if winner == Colour.BLUE:
            return -1
        elif winner == Colour.RED:
            return 1
        return 0

    @staticmethod
    def evaluateBoard2(boardState: BoardState, player: bool, turn: int) -> float:
        """
            Evaluates the board state by finding the shortest path via
            BFS, and how many tiles within it are currently occupied by
            the player
        """
        red_path_length = bfs(boardState, False)
        blue_path_length = bfs(boardState, True)

        colour = Colour.BLUE if player else Colour.RED
        colour_opp = Colour.RED if player else Colour.BLUE
        placed = 0
        placed_opp = 0

        game_progress_factor = 1 - (turn / 123)
        red_heuristic = (1 / (red_path_length)) * game_progress_factor
        blue_heuristic = (1 / (blue_path_length)) * game_progress_factor
        heuristic = red_heuristic - blue_heuristic

        return heuristic

def getDefensiveMoves(board: BoardState):
    directions = [-11, -10, 1, 11, 10, -1]

    moves = set({})
    if board.player:
        occupied = board.red_occupied
    else:
        occupied = board.blue_occupied

    for address in occupied:
        for offset in directions:
            n = address + offset

            # Check if tile is unoccupied
            if 0 <= n < 121 and board.tiles[n] == '0':
                moves.add(n)

    move_list = list(moves)
    return move_list


def bfs(board: BoardState, player: bool):
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    if player: # BLUE PLAYER
        colour = Colour.BLUE.get_char()
        start_nodes = [(i, 0) for i in range(11) if board.tiles[i * 11] == colour]
        target_side = lambda x, y: y == 10
    else: # RED PLAYER
        colour = Colour.RED.get_char()
        start_nodes = [(0, j) for j in range(11) if board.tiles[j] == colour]
        target_side = lambda x, y: x == 10

    queue = deque([(x, y, [(x, y)]) for x, y in start_nodes])  # (x, y, path_so_far)
    visited = set(start_nodes)
    while queue:
        x, y, path = queue.popleft()

        # Check if we've reached the target side
        if target_side(x, y):
            return len(path)

        # Explore neighbours
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and if the neighbour doesn't belong to the opponent
            if 0 <= nx < 11 and 0 <= ny < 11 and board.tiles[nx * 11 + ny] == colour and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [(nx, ny)]))


    return -1

def chooseBestPath(board: BoardState, player: bool, valid_actions: list[int]):
    n = 11 # Size of board

    # Directions for hex neighbours
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

    # Priority queue for Dijkstra's algorithm
    pq = []
    heapq.heapify(pq)

    # Distance map: cost to reach each tile
    dist = [[math.inf] * n for _ in range(n)]

    # Path map: to reconstruct the path
    prev = [[None] * n for _ in range(n)]

    # Initialize starting nodes
    if player:
        colour = Colour.BLUE.get_char()
        start_nodes = [(i, 0) for i in range(n) if board.tiles[11 * i] in ('0','B')]
        target_side = lambda x, y: y == 10
    else:
        colour = Colour.RED.get_char()
        start_nodes = [(0, j) for j in range(n) if board.tiles[j] in ('0', 'R')]
        target_side = lambda x, y: x == 10

    for x, y in start_nodes:
        cost = 0 if board.tiles[x * 11 + y] == colour else 1  # Cost is 0 if it's player's tile, 1 otherwise
        heapq.heappush(pq, (cost, x, y))
        dist[x][y] = cost

    # Dijkstra's main loop
    while pq:
        current_cost, x, y = heapq.heappop(pq)

        # Stop if we reached the target side
        if target_side(x, y):
            # Reconstruct the path
            path = []
            while (x, y) is not None:
                # Only care about the unoccupied tiles
                if board.tiles[x * 11 + y] == '0':
                    path.append(x * 11 + y)
                if prev[x][y] is None:
                    break
                x, y = prev[x][y]
            return path if len(path) else []

        # Skip if we've already found a better way to this node
        if current_cost > dist[x][y]:
            continue

        # Explore neighbours
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds
            if 0 <= nx < n and 0 <= ny < n:
                # Calculate cost
                if board.tiles[nx * 11 + ny] == colour:
                    next_cost = current_cost  # No additional cost for player's tile
                elif board.tiles[nx * 11 + ny] == '0':
                    next_cost = current_cost + 1  # Add cost for empty tile
                else:
                    continue  # Skip opponent's tiles

                # Relaxation step
                if next_cost < dist[nx][ny]:
                    dist[nx][ny] = next_cost
                    prev[nx][ny] = (x, y)
                    heapq.heappush(pq, (next_cost, nx, ny))

    # No path found so random move
    return []

def selectMove(offensive_moves: list[int], defensive_moves: list[int], valid_actions: list[int], offensive_threshold: float, defensive_threshold: float):
    if offensive_threshold + defensive_threshold >= 1:
        raise ValueError('The sum of offensiveThreshold and defensiveThreshold must be less than 1.')

    ideal_moves = list(set(offensive_moves) & set(defensive_moves))

    num = randint(1, 100) / 100

    if ideal_moves:
        # overlap between offensive and defensive moves
        if num < offensive_threshold + defensive_threshold:
            move = choice(ideal_moves)
        else:
            move = choice(valid_actions)

        if move in offensive_moves:
            offensive_moves.remove(move)
        return move
    else:
        # no overlap, handle separately
        thresholds = [
            (offensive_threshold, offensive_moves),
            (defensive_threshold, defensive_moves),
            (1, valid_actions)
        ]


        for threshold, moves in thresholds:
            if num < threshold and moves:
                move = choice(moves)
                if move in offensive_moves:
                    offensive_moves.remove(move)
                return move
            num -= threshold

    raise ValueError('No valid moves found.')
