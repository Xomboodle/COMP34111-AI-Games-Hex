# pylint: disable=invalid-name, superfluous-parens, too-many-arguments, line-too-long,
"""Script for generating bootstrap data by playing agents against each other."""
import argparse
import importlib
import json
import math
import random
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
        self.verbose = verbose

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

            print(f'\tturn {self.turn}', end='\r')

            # boardCopy = copy.deepcopy(self.board)
            # turnCopy = self.turn
            # playerCopy = copy.deepcopy(self.players)

            # playerBoard = copy.deepcopy(self.board)
            playerBoard = self.board

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

            if (self.verbose):
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
        result = self._end_game(endState)
        print(f'\t{result["winner"]} won in {self.turn} turns')

        # save results
        if (os.path.exists(DATA_PATH) is False):
            os.makedirs(DATA_PATH)

        winner = result['winner']
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
                else:
                    if (player.name == self.player1.name):
                        self.data[board]['payoff'] -= 1
                    else:
                        self.data[board]['payoff'] += 1
        # ---

        return result

def selfPlay(numTourneys=1, numRounds=1, players=('chump', 'chump'), hyperparamaterise=False, verbose=False):
    """Iteratively play games between agents to generate data."""
    step = 0.1
    checkpoint = step

    options = {
        'chump': 'agents.DefaultAgents.NaiveAgent NaiveAgent',
        'monkey': 'agents.Group27.MCTSAgent MCTSAgent',
    }

    p1Path, p1Class = options[players[0]].split(" ")
    p2Path, p2Class = options[players[1]].split(" ")
    p1 = importlib.import_module(p1Path)
    p2 = importlib.import_module(p2Path)

    best = (0, None)

    if (hyperparamaterise):
        grid = []
        temp_step = 0.2
        x_values = [round(i * step, 2) for i in range(1, int(0.95 / temp_step))]
        y_values = [round(i * step, 2) for i in range(1, int(0.95 / temp_step))]
        for x in x_values:
            for y in y_values:
                if (x + y > 0.95):
                    break
                grid.append((x, y))
        print(f'Running a grid search with {len(grid)} hyperparameter combinations', end='\n\n')
        numTourneys = len(grid)
    else:
        grid = []

    numOfOpponentsPerTournament = max(1, int(math.sqrt(len(grid))))

    for tourney in range(numTourneys):
        totalWins = 0

        if (grid):
            # grid search for p1
            p1Args = { 'offensive_threshold': grid[tourney][0], 'defensive_threshold': grid[tourney][1], 'debug': False } if (players[0] == 'monkey') else {}
            print(p1Args)
        else:
            p1Args = {}

        print(f'[{tourney+1}] {players[0]} (Alice) vs {players[1]} (Bob)')

        # random sample for p2
        totalBouts = 0
        for p2Sample in range(1, numOfOpponentsPerTournament + 1):

            if (grid):
                # random sample for p2
                if (players[1] == 'monkey'):
                    p2ArgsRaw = random.choice(grid)
                    p2Args = { 'offensive_threshold': p2ArgsRaw[0], 'defensive_threshold': p2ArgsRaw[1], 'debug': False }
                    print(p2Args)
                else:
                    p2Args = {}
            else:
                p2Args = {}

            startTime = time.time()

            game = CustomGame(
                player1=Player(
                    name='Alice',
                    agent=getattr(p1, p1Class)(Colour.RED, **p1Args),
                ),
                player2=Player(
                    name='Bob',
                    agent=getattr(p2, p2Class)(Colour.BLUE, **p2Args),
                ),
                verbose=False,
                silent=True,
            )

            wins = 0
            for bout in range(numRounds):
                totalBouts += 1

                game.reset()
                results = game.run()
                win = results["winner"] == "Alice"
                if win:
                    wins += 1

                percentage = totalBouts / (numRounds * numOfOpponentsPerTournament)
                if (verbose and percentage >= checkpoint):
                    print(f'\t{percentage * 100:.0f}%')
                    checkpoint += step
            totalWins += wins

            # save data
            with open(os.path.join(DATA_PATH, f'{players[0]}-v-{players[1]}.json'), 'w', encoding='utf-8') as f:
                json.dump(game.data, f, ensure_ascii=False, indent=4, separators=(',', ':'))
            # reformat
            with open(os.path.join(DATA_PATH, f'{players[0]}-v-{players[1]}.json'), 'r', encoding='utf-8') as f:
                data = f.read()
            data = data.replace('\n                ', '')
            data = data.replace('\n            ]', ']')
            with open(os.path.join(DATA_PATH, f'{players[0]}-v-{players[1]}.json'), 'w', encoding='utf-8') as f:
                f.write(data)

        endTime = time.time()
        elapsedTime = endTime - startTime

        if (wins > best[0]):
            best = (wins, p1Args)

        print(f'Played {numRounds} game{"s" if numRounds > 1 else ""}')
        if (p2Sample > 1):
            print(f'Against {p2Sample} opponents')
        print(f'Won {totalWins} ({totalWins / (numRounds * p2Sample) * 100:.0f}%)')
        print(f'Took {elapsedTime:.2f} seconds', end='\n\n')

    if (hyperparamaterise):
        print('DONE!')
        print(f'Best hyperparameters: {best[1]} ({best[0]} wins)')

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Generate bootstrap data with self-play.')
    parser.add_argument(
        'p1',
        type=str,
        nargs='?',
        default='monkey',
        help='P1 agent.',
    )
    parser.add_argument(
        'p2',
        type=str,
        nargs='?',
        default='chump',
        help='P2 agent.',
    )
    parser.add_argument(
        'numTourneys',
        type=int,
        nargs='?',
        default=1,
        help='Number of games to play.',
    )
    parser.add_argument(
        'numRounds',
        type=int,
        nargs='?',
        default=2,
        help='Number of matches to play per game.',
    )
    parser.add_argument(
        'hyperparamaterise',
        type=bool,
        nargs='?',
        default=False,
        help='Should monkey arguments be hyperparamaterised?',
    )
    args = parser.parse_args()
    selfPlay(args.numTourneys, args.numRounds, players=(args.p1, args.p2), hyperparamaterise=args.hyperparamaterise, verbose=True)
