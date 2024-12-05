"""TODO"""
from __future__ import annotations  # Enables forward references

from src.Board import Board
from src.Move import Move
from math import inf, log, sqrt
from agents.Group27.utils.BoardState import BoardState
from operator import truth
from threading import Thread
from multiprocessing.connection import Connection, Pipe
       

def get_moves(board: Board, turn: int) -> list:
    moves = []
    for x, line in enumerate(board.tiles):
        for y,tile in enumerate(line):
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
    def __init__(self, max_threads : int):
        super().__init__()
        self.max_threads = max_threads
        # self.pipes = pipes
        # setup thread to listen on connections
        # database_listener = Thread(target=self.listener, name="Database Listener")
        # database_listener.start() # we could move this out??

        self.updated_data = {}
        # list of node_hashes with value, visits
        self.nodes_to_add = []
        # list of parent_hash and action
        

    def listener(self, pipes):
        while True:
            i = 0
            while i < self.max_threads:
                try:
                    conn = pipes[i][0]
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
                        elif command == "STOP":
                            return
                except (EOFError, OSError) as e:
                    pipes[i] = Pipe()
                except IndexError:
                    i +=1
                    continue
                i += 1

    def update_from_data(self):
        """ Updates the entire tree with data acquired from last run """
        # all_pipes_free = False
        # while not(all_pipes_free):
        #     all_pipes_free = True
        #     for i in range(self.max_threads):
        #         conn = self.pipes[i][0]
        #         if conn.poll():
        #             all_pipes_free = False

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

def hash_to_node(node_hash : str) -> Node:
    state = BoardState()
    pointer = 0
    while pointer < len(node_hash):
        action = int(node_hash[pointer:pointer+3])
        state.make_move_address(action)
        pointer += 3
    return Node(state)