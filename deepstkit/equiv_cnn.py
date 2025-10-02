# deepstkit/equiv_cnn.py

import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantImageEncoder(nn.Module):
    def __init__(self, D=256):
        """
        E(2)-Equivariant CNN for image feature extraction.
        
        Args:
            D (int): The dimension of the output invariant feature 
      vector.
        """
        super(EquivariantImageEncoder, self).__init__()

        # Define the group of rotations we want to be equivariant to.
        # N=8 means we consider rotations by 360/8 = 45 degrees.
        self.r2_space = gspaces.Rot2dOnR2(N=8)

        # Define the feature type for the input image.
        # An RGB image has 3 channels, each is a trivial representation (it doesn't change with rotation).
        in_type = enn.FieldType(self.r2_space, [self.r2_space.trivial_repr] * 3)

        # Define the architecture of the CNN.
        # We will use a sequence of equivariant blocks.
        self.model = nn.Sequential(
            # Block 1
            enn.R2Conv(in_type, enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 16), kernel_size=7, padding=3),
            enn.ReLU(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 16), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 16), kernel_size=2),

            # Block 2
            enn.R2Conv(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 16), enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 32), kernel_size=5, padding=2),
            enn.ReLU(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 32), inplace=True),enn.PointwiseMaxPool(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 32), kernel_size=2),

            # Block 3
            enn.R2Conv(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 32), enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 64), kernel_size=5, padding=2),
            enn.ReLU(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 64), inplace=True),
            
            # Group Pooling to produce invariant features
            # This averages the features over the rotation group, making the output invariant.
            enn.GroupPooling(enn.FieldType(self.r2_space, [self.r2_space.regular_repr] * 64)),)

        # Final linear layer to project the invariant features to thedesired dimension D
        # The input to this layer is 64 because the GroupPooling outputs 64 channels of invariant features.
        self.final_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, D)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W]
        
        Returns:
            torch.Tensor: Output tensor of shape [B, D]
        """
        # Wrap the input tensor in a GeometricTensor
        x_geom = enn.GeometricTensor(x, enn.FieldType(self.r2_space, [self.r2_space.trivial_repr] * 3))
        
        # Pass through the equivariant CNN
        features_equiv = self.model(x_geom)
        
        # Unwrap the GeometricTensor to a standard torch.Tensor for the final MLP
        features_inv = features_equiv.tensor
        
        # Pass through the final MLP to get the embedding
        embedding = self.final_mlp(features_inv)
        
        return embedding