from agents.Group27.mcts.Node import Node, get_moves
from agents.Group27.utils.Heuristics import Heuristics
from src.Board import Board
from src.Move import Move
from src.Game import Game
from src.Colour import Colour
from math import inf, log, sqrt
import random
import copy
from operator import truth


def apply_move(node: Node, move : Move) -> tuple[Board, bool]:
    board = copy.deepcopy(node.state)
    player = node.player
    if not(move.x == -1 and move.y == -1): # if no swap
        board.set_tile_colour(move.x, move.y, node.get_player_colour())
        player = not(node.player)
    return board, player

class MCTS:
    '''TODO'''

    def __init__(self,max_simulations, *hyp_params):
        self.hyp_params = hyp_params
        self.max_simulations = max_simulations
        self.root_node = Node(Board(),False, turn=1)
        self.states = {str(self.root_node.state)+"1" : self.root_node}
        # explore root node
        while self.root_node.untried_actions:
            self.expand(self.root_node)

    def import_states():
        '''TODO'''
    
    def export_states():
        '''TODO'''
    
    def search(self, turn:int, heuristics: list, board: Board, previous_move : Move):

        self.current_state = copy.deepcopy(board) ## should be aight
        try:
            self.current_node = self.states[str(self.current_state)+str(turn)]
        except KeyError:
            # current node is the last turn in the this state
            self.states[str(self.current_state)+str(turn)] = Node(self.current_state,
                                                                  not(self.current_node.player),
                                                                  self.current_node,
                                                                  turn,
                                                                  previous_move)
            self.current_node = self.states[str(self.current_state)+str(turn)]

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
        self.current_node = self.states[str(apply_move(self.current_node, best_move)[0])+str(turn+1)]
        return best_move
        
    def select(self, node : Node) -> Node:
        """Select the child with the highest UCB1 value"""
        while node.untried_actions == [] and node.children:  # If all actions are tried and children exist
            node = max(node.children, key=lambda child: child.ucb())  # Select the best child by UCB1
        return node
    
    def expand(self, node: Node) -> Node:
        """Expand the node by trying one of the untried actions."""
        move = node.untried_actions.pop()
        turn = node.turn + 1
        next_state, player = apply_move(node, move)  # Simulate the state transition

        child_node = Node(next_state, player=player, parent=node, turn=turn, action=move)
        if turn == 4:
            pass
        self.states[str(next_state) + str(turn)] = child_node
        node.children.append(child_node)
        return child_node
    
    def simulate(self, node : Node, depth_limit : int = 50) -> float:
        """Simulate a random playout from the current node's state."""
        # node = copy.deepcopy(node)
        current_state = node.state
        player_colour = node.get_player_colour()
        depth_count = 0
        player = node.player
        # truth from operators is used as it is faster than bool(actions)
        # truth evaluates if there are any actions
        while truth(node.actions) and depth_count < depth_limit:  # Until the game reaches a terminal state
            actions = node.actions
            move = random.choice(actions)  # Pick a random action
            current_state,player = apply_move(node,move)  # Apply the action
            node = Node(current_state, player,turn=node.turn+1, parent=node, action=move)
            depth_count += 1
        return 0 #Heuristics.evaluateBoard(current_state, player, [0.1,0.7,0.1, 0.1]) # Return the reward of the terminal state

    def back_propagate(self, node: Node, reward : float):
        """Backpropagate the reward through the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self):
        """Return the best action based on the visit counts."""
        return max(self.current_node.children, key=lambda child: child.visits).action
