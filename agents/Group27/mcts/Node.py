"""TODO"""
from __future__ import annotations  # Enables forward references

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from math import inf, log, sqrt
import copy

def get_moves(board: Board, turn: int) -> list:
    moves = []
    for y, line in enumerate(board.tiles):
        for x,tile in enumerate(line):
            result = False
            if (0 <= x < board.size) and (0 <= y < board.size):
                # is in bound?
                tile = board.tiles[x][y]
                result = tile.colour is None
            elif x == -1 and y == -1 and turn == 2:
                # is a swap move?
                result = True
            else:
                result = False
            if result:
                moves.append(Move(x,y))
    return moves

class Node:
    """TODO"""

    def __init__(self, state: Board, player: bool, parent: Node | None = None, turn : int = 0, action : Move = None):
        self.state = state
        self.turn = turn
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.value = 0.0
        self.player = player # false for 1, true for 2

        # if self.parent:
        #     self.untried_actions = copy.deepcopy(self.parent.actions)
        #     self.untried_actions.remove(action)
        # else:
        #     self.untried_actions = get_moves(self.state, self.turn)    

        # self.actions = self.untried_actions.copy()    
        self.untried_actions = get_moves(self.state, self.turn)
        self.actions = self.untried_actions.copy()
        self.action = action

    def add_child(self, child_state: Board) -> Node:
        """Update children with new child state"""
        child = Node(child_state, not(self.player), self)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        """Update the node with the result of a simulation"""
        self.visits += 1
        self.value += result

    def get_player_colour(self):
        return Colour.BLUE if self.player else Colour.RED

    def ucb(self, exporlation_factor=1.4):
        """UCB calculation"""
        if self.visits == 0:
            return inf
        return self.value/self.visits + exporlation_factor * sqrt(log(self.parent.visits) / self.visits)
