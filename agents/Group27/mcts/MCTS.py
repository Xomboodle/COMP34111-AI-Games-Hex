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
from threading import Thread

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
        # has_ended = bfsFinished(boardstate_to_board(state), False)
    # h = Heuristics.evaluateBoard2(boardstate_to_board(state), False,state.turn)
    h = Heuristics.evaluate_basic(boardstate_to_board(state), False)
    # print(boardstate_to_board(node.state).print_board())
    # print(h)
    return  h #Heuristics.evaluateBoard(current_state, player, [0.1,0.7,0.1, 0.1]) # Return the reward of the terminal state


def back_propagate(tree : Tree, node: Node, reward : float):
    """Backpropagate the reward through the tree."""
    tree.back_propagate(node,reward)


class Tree:
    def __init__(self):
        self.root_node = Node(BoardState())

        self.nodes = {self.root_node.get_hash() : self.root_node}

    def get(self, node_hash : str) -> Node:
        return self.nodes[node_hash]

    def get_node_children(self):
        """Returns a list of the nodes children TODO"""
        pass
    
    def get_node_ucb(self, node: Node, exploration_factor : float = 1.4) -> float:
        if node.visits == 0:
            return inf
        parent = self.get(node.state.last_move_sequence)
        if parent.visits == 0:
            pass
        return node.value/node.visits + exploration_factor * sqrt(log(parent.visits) / node.visits)

    def get_best_child(self, node: Node, maximise : bool = True) -> Node:

        # h = max(node.next_states, key=lambda child_hash: self.get(child_hash).visits)
        if maximise:
            #red
            h = max(node.next_states, key=lambda child_hash: self.get(child_hash).value/self.get(child_hash).visits)
        else:
            #blue
            h = min(node.next_states, key=lambda child_hash: self.get(child_hash).value/self.get(child_hash).visits)
        return self.get(h)
    
    def add(self, parent_hash : str, node_action : int) -> None:
        """Adds a node following the given action
        Action give is an address"""
        parent = self.get(parent_hash)
        if node_action not in parent.untried_actions:
            pass
        parent.untried_actions.remove(node_action) # if this fails what the fuck
        next_state = parent.state.copy()
        next_state.make_move_address(node_action)
        new_node = Node(next_state)
        self.nodes[new_node.get_hash()] = new_node
        return new_node
    
    def back_propagate(self, node : Node, reward : float):
        """Backpropagate the reward through the tree."""
        while node.get_hash() != self.root_node.get_hash():
            node.update(reward)
            node = self.get(node.parent_hash)
        node.update(reward)



class Node:
    """Node built to decouple each node from each other"""
    def __init__(self,state : BoardState):
        self.state = state
        self.value = 0
        self.visits = 0

        self.next_states : list[str] = [] # move sequences to get better hashing
        for i in self.state.valid_actions:
            self.next_states.append(str(self.state) + str(i).zfill(3))

        self.untried_actions = self.state.valid_actions.copy()
        self.is_terminal = not(truth(self.state.valid_actions))

    def get_hash(self):
        return str(self.state)
    
    def update(self, value : float, visits : int = 1):
        self.value += value
        self.visits += visits

    @property
    def parent_hash(self):
        return self.state.last_move_sequence


## THREADING

class DatabaseTree(Tree):
    def __init__(self, pipes : list[list[Connection, Connection]], max_threads : int):
        super().__init__()
        self.max_threads = max_threads
        self.pipes = pipes
        # setup thread to listen on connections
        database_listener = Thread(target=self.listener, name="Database Listener")
        database_listener.start()

        self.updated_data = {}
        # list of node_hashes with value, visits
        self.nodes_to_add = []
        # list of parent_hash and action
        

    def listener(self):
        while True:
            i = 0
            while i < self.max_threads:
                try:
                    conn = self.pipes[i][0]
                    if conn.poll():
                        #input to read
                        message = conn.recv() # should only be hashes
                        # message must contain
                        # [command, input]
                        command, data = message
                        if command == "GET":
                            if data in self.nodes:
                                conn.send(self.get(data))
                            else:
                                conn.send(None)
                        elif command == "UPDATE":
                            ## data format [hash, reward]
                            h = data[0]
                            if h in self.updated_data:
                                self.updated_data[h][0] += data[1]
                                self.updated_data[h][1] += 1
                            else:
                                self.updated_data[h] = [data[1],1]
                        elif command == "ADD":
                            self.nodes_to_add.append(data)
                except (EOFError, OSError) as e:
                    self.pipes[i] = Pipe()
                except IndexError:
                    i +=1
                    continue
                i += 1

    def update_from_data(self):
        """ Updates the entire tree with data acquired from last run """
        all_pipes_free = False
        while not(all_pipes_free):
            all_pipes_free = True
            for i in range(self.max_threads):
                conn = self.pipes[i][0]
                if conn.poll():
                    all_pipes_free = False

        for n in self.nodes_to_add:
            if n[1] in self.get(n[0]).untried_actions:
                self.add(n[0], n[1])

        for node_hash in self.updated_data:
            data = self.updated_data[node_hash]
            updated_value = data[0]
            updated_visits = data[1]
            node = self.get(node_hash)
            node.visits += updated_visits
            node.value += updated_value

        self.nodes_to_add = []
        self.updated_data = {}
        self.pipes = []




class TreeCache(Tree):
    def __init__(self, conn : Connection):
        self.conn = conn
        self.nodes : dict[str, Node] = {}
        self.root_node = self.get("")
    
    def get(self, node_hash):
        if node_hash in self.nodes:
            return self.nodes[node_hash]
        self.conn.send(["GET",node_hash])
        node = self.conn.recv()
        if node is None:
            raise KeyError("Node does not exist in database")
        self.nodes[node_hash] = node
        return self.nodes[node_hash]
    
    def convert_nodes_to_dict(self) -> dict[str, dict]:
        out_dict = {}
        for node_hash in self.nodes:
            node = self.nodes[node_hash]

            # untried_array = np.array(node.untried_actions)
            obj = { #"state": node.state,
                   "value": node.value,
                   "visits": node.visits,
                #    "untried_actions": untried_array
            }
            out_dict[node_hash] = obj
        
        return out_dict
    
    def back_propagate(self, node : Node, reward : float):
        """Backpropagate the reward through the tree."""
        while node.get_hash() != self.root_node.get_hash():
            node.update(reward)
            self.conn.send(["UPDATE",[node.get_hash(),reward]])
            node = self.get(node.parent_hash)
        node.update(reward)
        self.conn.send(["UPDATE",[node.get_hash(),reward]])

    def add(self, parent_hash : str, node_action : int) -> None:
        super().add(parent_hash, node_action)
        self.conn.send(["ADD", [parent_hash, node_action]])

    

# class BranchedSearcher(Searcher):
#     """ Is the instance of searcher that is used by external threads"""
#     def __init__(self, conn : Connection):
#         self.tree = TreeCache(conn)

#         self.max_simulations = 300
#         self.max_depth = 200

#         self.last_move_sequence = ""
#         self.current_move_sequence = ""
#         self.player = False # True for blue, False for Red

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
    
def hash_to_node(node_hash : str) -> Node:
    state = BoardState()
    pointer = 0
    while pointer < len(node_hash):
        action = int(node_hash[pointer:pointer+3])
        state.make_move_address(action)
        pointer += 3
    return Node(state)