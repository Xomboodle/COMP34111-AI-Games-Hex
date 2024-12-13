from agents.Group27.cnn.HeuristicModel import HeuristicModel
from agents.Group27.cnn.PolicyModel import PolicyModel
from agents.Group27.mcts.Tree import Node, get_moves
from agents.Group27.utils.BoardState import BoardState, board_to_boardstate, boardstate_to_board
from agents.Group27.utils.Heuristics import *
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from math import inf, log, sqrt
import copy
from operator import truth

from multiprocessing.connection import Connection, Pipe, wait
from multiprocessing import Pool, Process, Queue
from agents.Group27.mcts.Tree import Node, Tree, TreeCache, DatabaseTree
from threading import Thread

import numpy as np
import random


class Searcher:
    def __init__(self, offensive_threshold: float, defensive_threshold: float, policyModel: PolicyModel, heuristicModel: HeuristicModel, debug: bool = False):
        self.tree = Tree()

        self.max_simulations = 500 if (policyModel is not None) else 1000
        self.max_depth = 200

        self.last_move_sequence = ""
        self.current_move_sequence = ""
        self.player = False # True for blue, False for Red

        self.offensive_threshold = offensive_threshold
        self.defensive_threshold = defensive_threshold

        self.policyModel = policyModel
        self.heuristicModel = heuristicModel

        self.debug = debug

    def search(self, turn:int, _heuristics: list, board: Board, previous_move : Move):
        _check = self.last_move_sequence

        self.last_move_sequence = self.current_move_sequence
        if previous_move is not None:
            previous_move_address = str(previous_move.x*board.size + previous_move.y).zfill(3)
        if turn == 1:
            self.player = False
            self.last_move_sequence = ""
            self.current_move_sequence = ""
        elif turn == 2:
            self.player = True
            # converts the previous move to an address
            self.last_move_sequence = ""
            self.current_move_sequence = previous_move_address
        elif turn == 3:
            if previous_move.x == -1:
                self.player = not(self.player)
            self.current_move_sequence += previous_move_address
        else:
            self.current_move_sequence += previous_move_address

        try:
            current_node = self.tree.get(self.current_move_sequence)
        except KeyError:
            # the node doesn't exist yet
            if previous_move.x*board.size + previous_move.y not in self.tree.get(self.last_move_sequence).untried_actions:
                t = self.tree.get(self.last_move_sequence)
                pass
            self.tree.add(self.last_move_sequence, previous_move.x*board.size + previous_move.y)
            current_node = self.tree.get(self.current_move_sequence)

        addr = search(self.tree, current_node.get_hash(),
                        self.offensive_threshold, self.defensive_threshold,
                        self.policyModel, self.heuristicModel,
                        not(self.player), self.max_simulations, self.max_depth)

        if addr < 0:
            # we have said swap
            self.player = not(self.player)

        # updates these with our move (presumes our move is always successful)
        self.last_move_sequence = self.current_move_sequence
        self.current_move_sequence = self.current_move_sequence + str(addr).zfill(3)

        # for h in current_node.next_states:
        #     print(self.tree.get(h).state.last_move ,self.tree.get(h).value/self.tree.get(h).visits)
        n = self.tree.get(self.current_move_sequence)
        if (self.debug):
            print(n.state.last_move, n.value/n.visits)

        return current_node.state.address_to_move(addr)


def search(
    tree: Tree, start_node_address: str,
    offensive_threshold: float, defensive_threshold: float,
    policyModel: PolicyModel, heuristicModel: HeuristicModel,
    maximise: bool, max_sims: int = 50, depth_limit: int = 200) -> int:
    """Searches the tree for the best child
    maximise - indicates if the search should maximise or minimise the heuristic

    Returns the address of the move that is best
    """
    current_node_address = start_node_address
    for _ in range(max_sims):
        node = select(tree, current_node_address)

        if node.untried_actions:
            expand(tree, node, policyModel)

        reward = simulate(node, offensive_threshold, defensive_threshold, heuristicModel, depth_limit)

        back_propagate(tree, node, reward)

    return tree.get_best_child(tree.get(start_node_address),maximise).state.last_move

def select(tree : Tree, node_address : str):
    """Select the child with the highest UCB1 value"""
    node = tree.get(node_address)
    if node is None:
        pass
    if node.untried_actions == []:
        pass
    while node.untried_actions == [] and not(node.is_terminal):  # If all actions are tried and children exist
        ucbs = [tree.get_node_ucb(tree.get(child_hash)) for child_hash in node.next_states]
        node_hash = max(node.next_states, key=lambda child_hash: tree.get_node_ucb(tree.get(child_hash)))
        node = tree.get(node_hash)
    return node

def expand(tree : Tree, node: Node, policyModel: PolicyModel) -> Node:
    """Expand the node by trying one of the untried actions."""
    skip = False
    if (policyModel is not None):
        policyOutput = policyModel(tensorfyBoard(node.state.tiles).float().unsqueeze(0)).squeeze()
        validMoves = torch.tensor(node.untried_actions)
        maskedPolicy = torch.full_like(policyOutput, float('-inf'))  # Mask all invalid moves
        maskedPolicy[validMoves] = policyOutput[validMoves]  # Keep only valid moves
        if (node.state.player):
            move = torch.argmin(maskedPolicy).item()
        else:
            move = torch.argmax(maskedPolicy).item()  # Get the highest valid move

        if (len(node.untried_actions) > 0 and move not in node.untried_actions):
            skip = True
    if (policyModel is None or skip):
        move = random.choice(node.untried_actions)

    return tree.add(node.get_hash(), move)

def simulate(
    node: Node,
    offensive_threshold: float, defensive_threshold: float,
    heuristicModel: HeuristicModel,
    depth_limit: int = 300) -> float:
    """Simulate a random playout from the current node's state."""

    state = node.state.copy()

    if (heuristicModel is None):
        # print(node.player)
        depth_count = 0
        # has_ended = dfsWinner(state)
        has_ended = False
        # Initial chooseBestMove returns path
        # Remove move selected on make_move_address
        # Check each iteration if opponent has placed in one of the moves
        # If so, recalculate
        current_path_p1 = []
        current_path_p2 = []

        while not(has_ended) and not(state.get_is_terminal()) and depth_count < depth_limit:  # Until the game reaches a terminal or deep state
            if len(current_path_p2) == 0 and state.player:
                current_path_p2 = chooseBestPath(state, state.player, state.valid_actions)#random.choice(node.state.valid_actions)  # Pick a random action
            if len(current_path_p1) == 0 and not(state.player):
                current_path_p1 = chooseBestPath(state, state.player, state.valid_actions)


            defensive_moves = getDefensiveMoves(state)
            if state.player and len(current_path_p2):
                move = selectMove(current_path_p2, defensive_moves, state.valid_actions, offensive_threshold, defensive_threshold)
                if move in current_path_p1:
                    current_path_p1 = []
            elif not(state.player) and len(current_path_p1):
                move = selectMove(current_path_p1, defensive_moves, state.valid_actions, offensive_threshold, defensive_threshold)
                if move in current_path_p2:
                    current_path_p2 = []
            else:
                break

            state.make_move_address(move)
            t = state.tiles[move]
            depth_count += 1
            has_ended = not(len(current_path_p1) or len(current_path_p2))

        h = Heuristics.evaluateBoard2(state, False,state.turn)
    else:
        h = heuristicModel(tensorfyBoard(state.tiles).float().unsqueeze(0)).item()
    # print(h)

    # print(boardstate_to_board(node.state).print_board())
    # print(h)
    return  h #Heuristics.evaluateBoard(current_state, player, [0.1,0.7,0.1, 0.1]) # Return the reward of the terminal state


def back_propagate(tree : Tree, node: Node, reward : float):
    """Backpropagate the reward through the tree."""
    tree.back_propagate(node,reward)




def branch_search(
    result_queue: Queue, conn: Connection, start_node_address: str,
    offensive_threshold: float, defensive_threshold: float,
    policyModel: PolicyModel, heuristicModel: HeuristicModel,
    maximise: bool, max_sims: int = 50, depth_limit: int = 200):
    # returns the nodes that were made/changed
    tree = TreeCache(conn)
    search(tree, start_node_address,
            offensive_threshold, defensive_threshold,
            policyModel, heuristicModel,
            maximise, max_sims, depth_limit)
    conn.close()
    # print(tree.nodes)
    # result_queue.put(tree.convert_nodes_to_dict())
    result_queue.close()

class MainSearcher(Searcher):
    def __init__(self, offensive_threshold: float, defensive_threshold: float, policyModel: PolicyModel, heuristicModel: HeuristicModel, debug: bool = False):
        super().__init__(offensive_threshold, defensive_threshold, policyModel, heuristicModel, debug)

        self.max_simulations = 500 if (policyModel is not None) else 1000
        self.max_depth = 200

        self.last_move_sequence = ""
        self.current_move_sequence = ""
        self.player = False # True for blue, False for Red

        self.max_threads = 7

        # self.pipes = [Pipe() for i in range(self.max_threads)]
        self.tree = DatabaseTree(self.max_threads)


    def search(self, turn:int, _heuristics: list, board: Board, previous_move : Move):
        _check = self.last_move_sequence

        self.last_move_sequence = self.current_move_sequence
        if previous_move is not None:
            previous_move_address = str(previous_move.x*board.size + previous_move.y).zfill(3)
        if turn == 1:
            self.player = False
            self.last_move_sequence = ""
            self.current_move_sequence = ""
        elif turn == 2:
            self.player = True
            # converts the previous move to an address
            self.last_move_sequence = ""
            self.current_move_sequence = previous_move_address
        elif turn == 3:
            if previous_move.x == -1:
                self.player = not(self.player)
            self.current_move_sequence += previous_move_address
        else:
            self.current_move_sequence += previous_move_address

        try:
            current_node = self.tree.get(self.current_move_sequence)
        except KeyError:
            # the node doesn't exist yet
            if previous_move.x*board.size + previous_move.y not in self.tree.get(self.last_move_sequence).untried_actions:
                t = self.tree.get(self.last_move_sequence)
                pass
            self.tree.add(self.last_move_sequence, previous_move.x*board.size + previous_move.y)
            current_node = self.tree.get(self.current_move_sequence)

        ### ^^^ Above copied

        pipes = [Pipe() for i in range(self.max_threads)]
        database_listener = Thread(target=self.tree.listener, args=([pipes]), name="Database Listener")
        database_listener.start() # we could move this out??

        result_queue = Queue() # TODO REMOVE
        # Spool up new processes
        args_set = [(
            result_queue, pipe[1], current_node.get_hash(),
            self.offensive_threshold, self.defensive_threshold,
            self.policyModel, self.heuristicModel,
            not(self.player), self.max_simulations, self.max_depth
        ) for pipe in pipes]
        pool = [Process(target=branch_search, args=args) for args in args_set]
        for process in pool:
            process.start()
        for process in pool:
            process.join(timeout=25)

        for pipe in pipes:
            pipe[1].send(["STOP",0])
        database_listener.join()



        # results : list[dict[str,Node]] = pool.starmap(branch_search,args)
        # self.pipes = []

        ## combine trees

        self.tree.update_from_data()

        addr = self.tree.get_best_child(self.tree.get(current_node.get_hash()),not(self.player)).state.last_move

        if addr < 0:
            # we have said swap
            self.player = not(self.player)

        # updates these with our move (presumes our move is always successful)
        self.last_move_sequence = self.current_move_sequence
        self.current_move_sequence = self.current_move_sequence + str(addr).zfill(3)

        # for h in current_node.next_states:
        #     print(self.tree.get(h).state.last_move ,self.tree.get(h).value/self.tree.get(h).visits)
        n = self.tree.get(self.current_move_sequence)
        if (self.debug):
            print(n.state.last_move, n.value/n.visits)

        return current_node.state.address_to_move(addr)

import torch

def tensorfyBoard(boardString: str) -> torch.Tensor:
    boardString = boardString.replace('R', '1').replace('B', '2')
    boardRows = [boardString[i:i+11] for i in range(0, len(boardString), 11)]
    board = [list(map(int, row)) for row in boardRows]

    board = torch.tensor(board, dtype=torch.int)

    rStones = (board == 1).int()
    bStones = (board == 2).int()
    nStones = (board == 0).int()

    return torch.stack([rStones, bStones, nStones])
