from agents.Group27.mcts.Node import Node, get_moves
from agents.Group27.utils.BoardState import BoardState, board_to_boardstate, boardstate_to_board
from agents.Group27.utils.Heuristics import Heuristics
from src.Board import Board
from src.Move import Move
from src.Game import Game
from src.Colour import Colour
from math import inf, log, sqrt
import random
import copy

def apply_move(node: Node, move : Move) -> tuple[Board, bool]:
    board = copy.deepcopy(node.state)
    player = node.player
    if not(move.x == -1 and move.y == -1): # if no swap
        board.set_tile_colour(move.x, move.y, node.get_player_colour())
        player = not(node.player)
    return board, player

def make_hash(state : BoardState, turn : int) -> str:
    s = str(state)
    if turn == 2:
        s += "SWAP"
    else:
        s += "NOSWAP"
    return s

class MCTS:
    '''TODO'''

    def __init__(self,max_simulations, *hyp_params):
        self.hyp_params = hyp_params
        self.max_simulations = max_simulations
        self.root_node = Node(BoardState(),False, turn=1)
        # dictionary of nodes depending on the board state and the turn
        self.nodes = {make_hash(self.root_node.state, 1) : self.root_node}
        self.current_node = self.root_node
        self.current_state = None

        # explore root node can probably try without this
        while self.root_node.untried_actions:
            self.expand(self.root_node)

    def import_states(self):
        """TODO"""
        pass

    def export_states(self):
        """TODO"""
        pass

    def search(self, turn:int, _heuristics: list, board: Board, previous_move : Move):

        self.current_state = board_to_boardstate(board) # BoardState
        try:
            self.current_node = self.nodes[str(self.current_state)+str(turn)]
        except KeyError:
            # current node is the last turn in the this state
            h = make_hash(self.current_state, turn)
            self.nodes[h] = Node(  self.current_state,
                                    not(self.current_node.player),
                                    self.current_node,
                                    turn,
                                    previous_move)
            self.current_node = self.nodes[h]

        for _ in range(self.max_simulations):
            # selection

            node = self.select(self.current_node) ## either increment, turn 0 or 1
            # exploration

            if node.untried_actions:
                node = self.expand(node)

            # simulation

            reward = self.simulate(node)
            # back propagate
            self.back_propagate(node, reward)

        best_move = self.best_action()
        # self.current_node = self.nodes[make_hash(self.current_node.state,turn+1)]
        return best_move

    def select(self, node : Node) -> Node:
        """Select the child with the highest UCB1 value"""
        while node.untried_actions == [] and node.children:  # If all actions are tried and children exist
            node = max(node.children, key=lambda child: child.ucb())  # Select the best child by UCB1
        return node

    def expand(self, node: Node) -> Node:
        """Expand the node by trying one of the untried actions."""
        move = random.choice(node.untried_actions)
        child_node = node.make_move(move)
        self.nodes[make_hash(child_node.state,child_node.turn)] = child_node
        node.children.append(child_node)
        return child_node

    def simulate(self, node : Node, depth_limit : int = 50) -> float:
        """Simulate a random playout from the current node's state."""
        depth_count = 0
        while not(node.is_terminal) and depth_count < depth_limit:  # Until the game reaches a terminal or deep state
            move = random.choice(node.state.valid_actions)  # Pick a random action
            # TODO: use the policy network to pick the best move, instead of random rollouts
            node = node.make_move(move)
            depth_count += 1
        return Heuristics.evaluateBoard2(boardstate_to_board(node.state), self.current_node.player) #Heuristics.evaluateBoard(current_state, player, [0.1,0.7,0.1, 0.1]) # Return the reward of the terminal state

    def back_propagate(self, node: Node, reward : float):
        """Backpropagate the reward through the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self):
        """Return the best action based on the visit counts."""
        return max(self.current_node.children, key=lambda child: child.visits).action
