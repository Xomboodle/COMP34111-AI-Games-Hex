import json
import os

import requests

ROOT_PATH = os.path.join(os.path.dirname(__file__))

with open(os.path.join(ROOT_PATH, 'ids.json'), 'r', encoding='utf-8') as f:
    expertGames = json.load(f)

expertData = []

for index, game in enumerate(expertGames):
    print(f'game {index}/{len(expertGames)}', end='\r')

    url = f"https://littlegolem.net/servlet/sgf/{game['game_id']}/game{game['game_id']}.hsgf"
    response = requests.get(url, timeout=60)

    if (response.status_code == 200):
        content = response.text
        game['moves'] = content
        expertData.append(game)
    else:
        print(f"Failed to fetch file {game['game_id']}. Status code: {response.status_code}")

with open(os.path.join(ROOT_PATH, 'expertData.json'), 'w', encoding='utf-8') as f:
    json.dump(expertData, f, indent=4)
