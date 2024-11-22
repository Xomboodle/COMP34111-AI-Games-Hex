from __future__ import annotations  # Enables forward references
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from operator import truth

def board_to_boardstate(board : Board) -> BoardState:
    ''' Generates a boardstate from a board '''
    bs = BoardState(board.size)
    for y,line in enumerate(board.tiles):
        for x,tile in enumerate(line):
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

        #actions will be referred to by address (move.y*self.size + move.x) reduces faff
        # we can also store actions in one big list.
        self.valid_actions = list(range(self.size**2))
        # note swapping is not provided by the board state. this is by the game

    def make_move(self,move : Move, colour : Colour):
        ''' Performs a move on the board state'''
        address = self.move_to_address(move)        
        self.make_move_address(address, colour)

    def make_move_address(self, address : int, colour : Colour):
        self.tiles = self.tiles[:address] + colour.get_char() + self.tiles[address +1:]
        if address not in self.valid_actions:
            pass
        self.valid_actions.remove(address) ## this might be slow?

    def address_to_move(self, address : int) -> Move:
        '''Converts an address to a move'''
        return Move(address//self.size, address%self.size)

    def move_to_address(self, move : Move) -> int:
        '''Converts a move to an address'''
        return move.y*self.size + move.x
    
    def __str__(self) -> str:
        return self.tiles
    
    def copy(self) -> BoardState:
        bs = BoardState(self.size)
        bs.tiles = self.tiles
        bs.valid_actions = self.valid_actions.copy()
        return bs

    def get_is_terminal(self):
        return not(truth(self.valid_actions))

