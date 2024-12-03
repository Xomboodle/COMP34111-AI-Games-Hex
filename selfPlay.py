# pylint: disable=invalid-name, superfluous-parens, too-many-arguments, line-too-long,
"""Script for generating bootstrap data by playing agents against each other."""
import argparse
import importlib
import json
import time
import os
import sys
import copy

from src.Board import Board
from src.Colour import Colour
from src.EndState import EndState
from src.Player import Player
from src.Game import Game

DATA_PATH = os.path.join(os.path.dirname(__file__), 'agents', 'Group27', 'data')

class CustomGame(Game):
    """TODO"""

    def __init__(self, player1, player2, board_size = 11, logDest = sys.stderr, verbose = False, silent = False):
        super().__init__(player1, player2, board_size, logDest, verbose, silent)
        self.data = {}

    def reset(self):
        """TODO"""
        self._turn = 0
        self._board = Board(self.board.size)
        self.current_player = Colour.RED
        self._start_time = time.time()
        if self.has_swapped:
            self.players[Colour.RED], self.players[Colour.BLUE] = (
                    self.players[Colour.BLUE],
                    self.players[Colour.RED],
                )
        self.has_swapped = False

        self.logDest = open(os.devnull, 'w', encoding='utf-8')

    def _play(self) -> dict[str, str]:
        """
        Taken from Game, but modified for recording metrics.
        """
        endState = EndState.WIN
        opponentMove = None

        # ---
        # setup
        p1Boards: dict[str, tuple[int, int]] = {}
        p2Boards: dict[str, tuple[int, int]] = {}
        # ---

        while True:
            self._turn += 1
            currentPlayer: Player = self.players[self.current_player]
            playerAgent = currentPlayer.agent
            currentPlayer.turn += 1

            # boardCopy = copy.deepcopy(self.board)
            turnCopy = self.turn
            # playerCopy = copy.deepcopy(self.players)

            playerBoard = copy.deepcopy(self.board)

            start = time.time()
            m = playerAgent.make_move(self.turn, playerBoard, opponentMove)
            end = time.time()

            # assert boardCopy == self.board, "Board was modified, Possible cheating!"
            # assert turnCopy == self.turn, "Turn was modified, Possible cheating!"
            # assert playerCopy == self.players, "Players were modified, Possible cheating!"
            # assert end > start, "Move time is negative, Possible cheating!"

            # ---
            # track moves
            boards = p1Boards if (self.current_player == Colour.RED) else p2Boards
            stringiBoard = str(playerBoard)
            boards[stringiBoard] = (m.x, m.y)
            # ---

            print(self.board.print_board()) # displays the board

            currentPlayer.move_time += int(end - start)
            if currentPlayer.move_time > Game.MAXIMUM_TIME:
                endState = EndState.TIMEOUT
                break
            if self.is_valid_move(m, self.turn, self.board):
                self._make_move(m)
                opponentMove = m
            else:
                endState = EndState.BAD_MOVE
                break
            if self.board.has_ended(self.current_player):
                break

            self.current_player = Colour.opposite(self.current_player)

        # ---
        # save results
        winner = self.players[self.current_player].name
        for (player, boards) in [(self.player1, p1Boards), (self.player2, p2Boards)]:
            for (board, move) in boards.items():

                if (self.data.get(board) is None):
                    self.data[board] = {
                        'moves': [[0 for _ in range(self.board.size)] for _ in range(self.board.size)],
                        'payoff': 0,
                    }

                # policy
                self.data[board]['moves'][move[0]][move[1]] += 1

                # payoffs
                if (player.name == winner):
                    if (player.name == self.player1.name):
                        self.data[board]['payoff'] += 1
                    else:
                        self.data[board]['payoff'] -= 1
        # ---

        return self._end_game(endState)

def selfPlay(totalGames=1, verbose=False):
    """Iteratively play games between agents to generate data."""
    step = 0.1
    checkpoint = step

    startTime = time.time()

    p1Path, p1Class = 'agents.Group27.MCTSAgent MCTSAgent'.split(" ")
    p2Path, p2Class = 'agents.DefaultAgents.NaiveAgent NaiveAgent'.split(" ")

    # p1Path, p1Class = 'agents.DefaultAgents.NaiveAgent NaiveAgent'.split(" ")
    # p2Path, p2Class = 'agents.Group27.MCTSAgent MCTSAgent'.split(" ")

    p1 = importlib.import_module(p1Path)
    p2 = importlib.import_module(p2Path)

    game = CustomGame(
        player1=Player(
            name='Alice',
            agent=getattr(p1, p1Class)(Colour.RED),
        ),
        player2=Player(
            name='Bob',
            agent=getattr(p2, p2Class)(Colour.BLUE),
        ),
        verbose=False,
        silent=True,
    )

    wins = 0
    for i in range(totalGames):

        game.reset()
        results = game.run()
        win = results["winner"] == "Alice"
        if win:
            wins += 1

        percentage = (i+1) / totalGames
        if (verbose and percentage >= checkpoint):
            print(f'{percentage * 100:.0f}%')
            checkpoint += step

    endTime = time.time()
    elapsedTime = endTime - startTime

    print(f'Won {wins} / {totalGames}')
    print(f'Played {totalGames} games')
    print(f'Took {elapsedTime:.2f} seconds')

    # save data
    # with open(os.path.join(DATA_PATH, 'chump-v-chump.json'), 'w', encoding='utf-8') as f:
    #     json.dump(game.data, f, ensure_ascii=False, indent=4, separators=(',', ':'))

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Generate bootstrap data with self-play.')
    parser.add_argument(
        'numGames',
        type=int,
        nargs='?',
        default=2,
        help='Number of games to play.',
    )
    args = parser.parse_args()
    selfPlay(args.numGames, True)
