"""CNN Model for MCTS."""
import torch
import torch.nn as nn

class PolicyModel(nn.Module):
    """Convolutional neural network for predicting game outcomes based on the Hex board state."""

    def __init__(self, boardSize=11):
        """Initialise model."""
        super(PolicyModel, self).__init__()

        # Convolutional layers (feature extraction)
        #   - kernel_size: The size of the convolutional kernel
        #   - padding: The number of pixels to add to each side of the input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head
        self.convp = nn.Conv2d(64, 2, kernel_size=1)  # reduce to 2 channels
        self.fcp = nn.Linear(2 * boardSize * boardSize, boardSize * boardSize)

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

        # Policy head
        p = torch.relu(self.convp(x))
        p = p.view(p.size(0), -1)  # flatten
        p = self.fcp(p)  # output move logits

        return p
