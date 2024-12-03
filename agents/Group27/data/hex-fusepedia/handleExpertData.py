import json
import os
from typing import Any

from src.Board import Board
from src.Colour import Colour

ROOT_PATH = os.path.join(os.path.dirname(__file__), 'agents', 'Group27', 'data')

with open(os.path.join(ROOT_PATH, 'hex-fusepedia', 'expertData.json'), 'r', encoding='utf-8') as f:
    expertData = json.load(f)

ctoi = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
}

BWtoRB = {
    'W': Colour.RED,
    'B': Colour.BLUE,
}

stats: dict[str, dict[str, Any]] = {}

for game in expertData:
    print(f"game {game['game_id']}")

    board = Board()
    # print(board.print_board())

    data = game['moves'].strip('(); ').split(';')
    p1Wins = True if game['red_won_flg'] == 1 else False

    meta = data[1]
    for chunk in data[2:]:

        if ('resign' in chunk):
            break
        if ('swap' in chunk):
            continue

        player = BWtoRB[chunk[0]]
        (y, x) = (ctoi[chunk[2]], ctoi[chunk[3]])
        board.set_tile_colour(x, y, player)
        stringiBoard = board.print_board()
        # print(board.print_board())

        if (stats.get(stringiBoard) is None):
            stats[stringiBoard] = {
                'moves': [[0 for _ in range(11)] for _ in range(11)],
                'payoff': 0,
            }

        # policy
        stats[stringiBoard]['moves'][y][x] += 1

        # payoffs
        if (p1Wins):
            if (player == Colour.RED):
                stats[stringiBoard]['payoff'] += 1
            else:
                stats[stringiBoard]['payoff'] -= 1
        else:
            if (player == Colour.RED):
                stats[stringiBoard]['payoff'] -= 1
            else:
                stats[stringiBoard]['payoff'] += 1

# save data
with open(os.path.join(ROOT_PATH, 'expert-v-expert.json'), 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=4, separators=(',', ':'))
# reformat
with open(os.path.join(ROOT_PATH, 'expert-v-expert.json'), 'r', encoding='utf-8') as f:
    data = f.read()
data = data.replace('\n                ', '')
data = data.replace('\n            ]', ']')
with open(os.path.join(ROOT_PATH, 'expert-v-expert.json'), 'w', encoding='utf-8') as f:
    f.write(data)
