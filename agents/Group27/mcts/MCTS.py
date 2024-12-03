from __future__ import annotations  # Enables forward references

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from math import inf, log, sqrt
import copy
from agents.Group27.utils.BoardState import BoardState, board_to_boardstate, boardstate_to_board
from operator import truth
from agents.Group27.utils.Heuristics import Heuristics, bfsFinished

from multiprocessing.connection import Connection, Pipe, wait
from multiprocessing import Pool, Process, Queue
from agents.Group27.mcts.Tree import Node, Tree, TreeCache, DatabaseTree

import numpy as np
import random
       

class Searcher:
    def __init__(self):
        self.tree = Tree()
        
        self.max_simulations = 300
        self.max_depth = 200

        self.last_move_sequence = ""
        self.current_move_sequence = ""
        self.player = False # True for blue, False for Red
    
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

        addr = search(self.tree,current_node.get_hash(), not(self.player), self.max_simulations, self.max_depth)


        if addr < 0:
            # we have said swap
            self.player = not(self.player)

        # updates these with our move (presumes our move is always successful)
        self.last_move_sequence = self.current_move_sequence
        self.current_move_sequence = self.current_move_sequence + str(addr).zfill(3)

        # for h in current_node.next_states:
        #     print(self.tree.get(h).state.last_move ,self.tree.get(h).value/self.tree.get(h).visits)
        n = self.tree.get(self.current_move_sequence)
        print(n.state.last_move, n.value/n.visits)

        return current_node.state.address_to_move(addr)

        

def search(tree : Tree, start_node_address : str, maximise : bool, max_sims : int = 50, depth_limit : int = 200) -> int:
    """Searches the tree for the best child
    maximise - indicates if the search should maximise or minimise the heuristic

    Returns the address of the move that is best
    """
    current_node_address = start_node_address
    for _ in range(max_sims):
        node = select(tree, current_node_address)

        if node.untried_actions:
            expand(tree, node)

        reward = simulate(node, depth_limit)

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

def expand(tree : Tree, node: Node) -> Node:
    """Expand the node by trying one of the untried actions."""
    move = random.choice(node.untried_actions)
    return tree.add(node.get_hash(), move)

def simulate(node : Node, depth_limit : int = 200) -> float:
    """Simulate a random playout from the current node's state."""

    # for now version ignoring if the game has properly ended (the winner will be the same)
    # depth_count = 0
    # state = node.state.copy()
    # while truth(state.valid_actions) and depth_count < depth_limit: 
    #     move = random.choice(state.valid_actions)
    #     state.make_move_address(move)
    
    # return Heuristics.evaluate_basic(boardstate_to_board(state), False)
    # return Heuristics.evaluateBoard2(boardstate_to_board(state), False, state.turn)

    # \/\/\/\/ Version that uses bfs search and more accurate heuristic need to be faster
    # print(node.player)
    depth_count = 0
    state = node.state.copy()
    board = boardstate_to_board(state)
    player = node.state.player
    has_ended = bfsFinished(board, player)
    while not(has_ended) and truth(state.valid_actions) and depth_count < depth_limit:  # Until the game reaches a terminal or deep state
        move = random.choice(state.valid_actions)  # Pick a random action
        # TODO: use the policy network to pick the best move, instead of random rollouts
        state.make_move_address(move)
        depth_count += 1
        has_ended = bfsFinished(boardstate_to_board(state), False) # slow
    h = Heuristics.evaluateBoard2(boardstate_to_board(state), False,state.turn)
    # h = Heuristics.evaluate_basic(boardstate_to_board(state), False)
    # print(boardstate_to_board(node.state).print_board())
    # print(h)
    return  h #Heuristics.evaluateBoard(current_state, player, [0.1,0.7,0.1, 0.1]) # Return the reward of the terminal state


def back_propagate(tree : Tree, node: Node, reward : float):
    """Backpropagate the reward through the tree."""
    tree.back_propagate(node,reward)




def branch_search(result_queue : Queue,conn : Connection, start_node_address : str, maximise : bool, max_sims : int = 50, depth_limit : int = 200):
    # returns the nodes that were made/changed
    tree = TreeCache(conn)
    search(tree,start_node_address,maximise,max_sims,depth_limit)
    conn.close()
    # print(tree.nodes)
    # result_queue.put(tree.convert_nodes_to_dict())
    result_queue.close()

class MainSearcher(Searcher):
    def __init__(self):
        

        self.max_simulations = 5000
        self.max_depth = 200

        self.last_move_sequence = ""
        self.current_move_sequence = ""
        self.player = False # True for blue, False for Red

        self.max_threads = 8
        

        self.pipes = [Pipe() for i in range(self.max_threads)]
        self.tree = DatabaseTree(self.pipes, self.max_threads)


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
        self.pipes = [Pipe() for i in range(self.max_threads)]
        self.tree.pipes = self.pipes
        result_queue = Queue()
        # pool = Pool(self.max_threads)
        # Spool up new processes
        args_set = [(result_queue, pipe[1],current_node.get_hash(), not(self.player), self.max_simulations, self.max_depth) for pipe in self.pipes]
        pool = [Process(target=branch_search, args=args) for args in args_set]
        for process in pool:
            process.start()
        results : list[dict] = []
        for process in pool:
            process.join(timeout=25)
            # results.append(result_queue.get())

        
        # results : list[dict[str,Node]] = pool.starmap(branch_search,args)
        self.pipes = []

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
        print(n.state.last_move, n.value/n.visits)

        return current_node.state.address_to_move(addr)
    
