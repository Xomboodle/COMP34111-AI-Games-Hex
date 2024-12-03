"""CNN Model for MCTS."""
import torch
import torch.nn as nn

class HeuristicModel(nn.Module):
    """Convolutional neural network for predicting game outcomes based on the Hex board state."""

    def __init__(self, boardSize=11):
        """Initialise model."""
        super(HeuristicModel, self).__init__()

        # Convolutional layers (feature extraction)
        #   - kernel_size: The size of the convolutional kernel
        #   - padding: The number of pixels to add to each side of the input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Heuristic head
        self.convh = nn.Conv2d(64, 2, kernel_size=1)  # reduce to 2 channels
        self.fch = nn.Linear(2 * boardSize * boardSize, 1)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, board_size, board_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), representing win/loss probability.
        """

        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Heuristic head
        p = torch.relu(self.convh(x))
        p = p.view(p.size(0), -1)  # flatten
        p = self.fch(p)  # output move logits

        return p
