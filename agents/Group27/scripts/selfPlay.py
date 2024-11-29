"""Script for generating bootstrap data by playing agents against each other."""
import argparse
import subprocess
import time

def selfPlay(totalGames=1, verbose=False):
    """Iteratively play games between agents to generate data."""
    step = 0.1
    checkpoint = step

    startTime = time.time()

    for i in range(totalGames):
        try:
            # Command to execute
            command = 'python3 Hex.py -p1 "agents.Group27.LearningAgent LearningAgent" -p2 "agents.Group27.LearningAgent LearningAgent"'
            # XXX: this script works, but is horribly inefficient, as it relies on re-writing the dict every move
            # having a local instance running would be much better.

            # Run the command
            subprocess.run(command, shell=True, check=True, text=True, stderr=subprocess.PIPE)

        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")

        percentage = (i+1) / totalGames
        if (verbose and percentage >= checkpoint):
            print(f'{checkpoint * 100:.0f}%')
            checkpoint += step

    endTime = time.time()
    elapsedTime = endTime - startTime
    print(f'Played {totalGames} games')
    print(f'Took {elapsedTime:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bootstrap data with self-play.")
    parser.add_argument(
        "numGames",
        type=int,
        help="Number of games to play.",
    )
    args = parser.parse_args()
    selfPlay(args.numGames)
