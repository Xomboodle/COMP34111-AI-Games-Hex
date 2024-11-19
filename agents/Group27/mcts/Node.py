"""TODO"""
from __future__ import annotations  # Enables forward references

from src.Board import Board

class Node:
    """TODO"""

    def __init__(self, state: Board, parent: Node | None = None):
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_state: Board) -> Node:
        """Update children with new child state"""
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        """Update the node with the result of a simulation"""
        self.visits += 1
        self.value += result
