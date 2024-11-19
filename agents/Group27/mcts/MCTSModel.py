"""CNN Model for MCTS."""
import torch
import torch.nn as nn

class MCTSModel(nn.Module):
    """Convolutional neural network for predicting game outcomes based on the Hex board state."""

    def __init__(self):
        """Initialise model."""
        super(MCTSModel, self).__init__()

        board_size = 11

        # Convolutional layers (feature extraction)
        #   - kernel_size: The size of the convolutional kernel
        #   - padding: The number of pixels to add to each side of the input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers (decision making)
        self.fc1 = nn.Linear(64 * board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 1)

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

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output a probability
        return x
