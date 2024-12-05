from __future__ import annotations  # Enables forward references
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from operator import truth

def board_to_boardstate(board : Board) -> BoardState:
    ''' Generates a boardstate from a board '''
    bs = BoardState(board.size)
    for x,line in enumerate(board.tiles):
        for y,tile in enumerate(line):
            if tile.colour:
                bs.make_move(Move(x,y), tile.colour)
    return bs

def boardstate_to_board(bs : BoardState) -> Board:
    b = Board(bs.size)
    for addr in range(bs.size**2):
        if bs.tiles[addr] != "0":
            m = bs.address_to_move(addr)
            b.set_tile_colour(m.x,m.y, Colour.from_char(bs.tiles[addr]))
    return b

class BoardState():
    '''Simpler data storage of board

        Allows for faster updating of valid actions within the state
    '''
    def __init__(self, board_size=11):
        
        self.size = board_size
        self.tiles = "0" * self.size * self.size
        self.turn = 1
        self.player = False # False = RED, True = BLUE
        self.move_sequence = ""
        self.last_move_sequence = ""
        self.last_move = 0

        #actions will be referred to by address (move.y*self.size + move.x) reduces faff
        # we can also store actions in one big list.
        self.valid_actions = list(range(self.size**2))

    def make_move(self,move : Move, colour : Colour):
        ''' Performs a move on the board state'''
        address = self.move_to_address(move)        
        self.make_move_address(address, colour)

    def make_move_address(self, address : int, colour : Colour = None):
        colour = Colour.BLUE if self.player else Colour.RED
        self.last_move_sequence = self.move_sequence
        self.last_move = address
        self.move_sequence += str(address).zfill(3)
    
        if self.turn == 1:  
            self.tiles = self.tiles[:address] + colour.get_char() + self.tiles[address +1:]
            self.valid_actions.append(-1*self.size -1)
        elif self.turn == 2:
            if address >= 0:
                # remove when the swap action is not taken
                self.valid_actions.remove(-1*self.size -1)
                self.tiles = self.tiles[:address] + colour.get_char() + self.tiles[address +1:]
            else:
                # swap is taken
                self.player = not(self.player)
        else:
            self.tiles = self.tiles[:address] + colour.get_char() + self.tiles[address +1:]
        self.player = not(self.player)

        self.turn += 1
        # if address not in self.valid_actions:
        #     pass
        self.valid_actions.remove(address) ## this might be slow?

    def address_to_move(self, address : int) -> Move:
        '''Converts an address to a move'''
        return Move(address//self.size, address%self.size)

    def move_to_address(self, move : Move) -> int:
        '''Converts a move to an address'''
        return move.x*self.size + move.y
    
    def __str__(self) -> str:
        """Performs conversion into hash"""
        return self.move_sequence
    
    def copy(self) -> BoardState:
        bs = BoardState(self.size)
        bs.tiles = self.tiles
        bs.valid_actions = self.valid_actions.copy()
        bs.last_move_sequence = self.last_move_sequence
        bs.move_sequence = self.move_sequence
        bs.last_move = self.last_move
        bs.player = self.player
        bs.turn = self.turn
        return bs

    def get_is_terminal(self):
        return not(truth(self.valid_actions))
