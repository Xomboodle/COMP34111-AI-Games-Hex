"""Group 27 Agent"""
from random import choice
from time import time

# import torch

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# from agents.Group27.mcts.PolicyModel import PolicyModel
from agents.Group27.mcts.MCTS import Searcher,MainSearcher

class MCTSAgent(AgentBase):
    """
    Desribes an agent that implements an optimised MCTS

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

        # self.model: PolicyModel = PolicyModel()
        # self.model.load_state_dict(torch.load('agents/Group27/mcts/test.pth'))
        # self.model.eval()
        self.mcts = MainSearcher() # multithreaded search
        # self.mcts = Searcher() # single threaded

        self.total_search_time = 0
        self.search_count = 0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent move
        """

        ## self.model.make_heuristics(board)
        self.search_count += 1
        a = time()
        move = self.mcts.search(turn, [], board, opp_move)
        b = time()

        time_searched = b-a
        self.total_search_time += time_searched
        print("Time: " + str(time_searched) + " Average Time: " + str(self.total_search_time/self.search_count))
        print(f"Number of nodes {len(self.mcts.tree.nodes)}")

        return move

        # if turn == 2 and choice([0, 1]) == 1:
        if turn == 2:
            return Move(-1, -1)
        else:
            x, y = choice(self._choices)
            return Move(x, y)
