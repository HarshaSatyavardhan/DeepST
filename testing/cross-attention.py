#!/usr/bin/env python
"""
Cross-Attention Mechanism for Gene Expression and Image Embeddings
Combines gene expression data with histology image patches using cross-attention
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# Add DeepST to path
sys.path.append('/scratch/harsha.vasamsetti/DeepST')
from deepstkit.utils_func import read_10X_Visium
from deepstkit.his_feat import image_crop, image_feature


class CrossAttention(nn.Module):
    """
    Cross-Attention module for fusing gene expression and image embeddings.

    Args:
        gene_dim: Dimension of gene expression embeddings
        image_dim: Dimension of image embeddings
        hidden_dim: Hidden dimension for attention computation
        num_heads: Number of attention heads
    """
    def __init__(self, gene_dim, image_dim, hidden_dim=256, num_heads=8):
        super(CrossAttention, self).__init__()

        self.gene_dim = gene_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Linear projections for gene expression (queries)
        self.gene_to_q = nn.Linear(gene_dim, hidden_dim)

        # Linear projections for image features (keys and values)
        self.image_to_k = nn.Linear(image_dim, hidden_dim)
        self.image_to_v = nn.Linear(image_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, gene_emb, image_emb):
        """
        Forward pass of cross-attention.

        Args:
            gene_emb: Gene expression embeddings [batch_size, gene_dim]
            image_emb: Image embeddings [batch_size, image_dim]

        Returns:
            fused_emb: Fused embeddings [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, num_heads, 1, 1]
        """
        batch_size = gene_emb.size(0)

        # Project to query, key, value
        Q = self.gene_to_q(gene_emb)  # [batch_size, hidden_dim]
        K = self.image_to_k(image_emb)  # [batch_size, hidden_dim]
        V = self.image_to_v(image_emb)  # [batch_size, hidden_dim]

        # Reshape for multi-head attention
        # [batch_size, num_heads, 1, head_dim]
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)

        # Output projection with residual connection
        out = self.out_proj(attended)
        out = self.norm1(out + Q.squeeze(1).view(batch_size, self.hidden_dim))

        # Feed-forward network with residual connection
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)

        return out, attention_weights


class GeneExpressionEncoder(nn.Module):
    """
    Encoder for gene expression data.

    Args:
        input_dim: Number of genes
        hidden_dims: List of hidden layer dimensions
        output_dim: Output embedding dimension
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128):
        super(GeneExpressionEncoder, self).__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ImagePatchEncoder(nn.Module):
    """
    Encoder for image patches using ResNet.

    Args:
        resnet_type: Type of ResNet ('resnet18', 'resnet50', etc.)
        output_dim: Output embedding dimension
        pretrained: Whether to use pretrained weights
    """
    def __init__(self, resnet_type='resnet50', output_dim=128, pretrained=True):
        super(ImagePatchEncoder, self).__init__()

        # Load pretrained ResNet
        if resnet_type == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            resnet_dim = 512
        elif resnet_type == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            resnet_dim = 512
        elif resnet_type == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            resnet_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet type: {resnet_type}")

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Add projection layer
        self.projection = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Image tensor [batch_size, 3, 224, 224]
        Returns:
            embeddings: Image embeddings [batch_size, output_dim]
        """
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        return embeddings


class MultiModalFusionModel(nn.Module):
    """
    Complete model for fusing gene expression and image data via cross-attention.
    """
    def __init__(self,
                 n_genes,
                 gene_emb_dim=128,
                 image_emb_dim=128,
                 fusion_dim=256,
                 num_heads=8):
        super(MultiModalFusionModel, self).__init__()

        self.gene_encoder = GeneExpressionEncoder(
            input_dim=n_genes,
            hidden_dims=[512, 256],
            output_dim=gene_emb_dim
        )

        self.image_encoder = ImagePatchEncoder(
            resnet_type='resnet50',
            output_dim=image_emb_dim,
            pretrained=True
        )

        self.cross_attention = CrossAttention(
            gene_dim=gene_emb_dim,
            image_dim=image_emb_dim,
            hidden_dim=fusion_dim,
            num_heads=num_heads
        )

    def forward(self, gene_expr, image_patch):
        """
        Args:
            gene_expr: Gene expression data [batch_size, n_genes]
            image_patch: Image patches [batch_size, 3, 224, 224]

        Returns:
            fused_embeddings: Fused embeddings [batch_size, fusion_dim]
            gene_embeddings: Gene embeddings [batch_size, gene_emb_dim]
            image_embeddings: Image embeddings [batch_size, image_emb_dim]
            attention_weights: Cross-attention weights
        """
        # Encode gene expression
        gene_embeddings = self.gene_encoder(gene_expr)

        # Encode image patches
        image_embeddings = self.image_encoder(image_patch)

        # Fuse via cross-attention
        fused_embeddings, attention_weights = self.cross_attention(
            gene_embeddings, image_embeddings
        )

        return fused_embeddings, gene_embeddings, image_embeddings, attention_weights


class SpatialDataset(Dataset):
    """Dataset for spatial transcriptomics data."""

    def __init__(self, adata, transform=None):
        self.adata = adata
        self.transform = transform

        # Get gene expression matrix
        if hasattr(adata.X, 'toarray'):
            self.gene_expr = adata.X.toarray()
        else:
            self.gene_expr = adata.X

        self.slice_paths = adata.obs['slices_path'].values

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Get gene expression
        gene_expr = torch.FloatTensor(self.gene_expr[idx])

        # Load and transform image
        img_path = self.slice_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return gene_expr, image, idx


def load_and_prepare_data(data_path, save_path):
    """
    Load and prepare spatial transcriptomics data.

    Args:
        data_path: Path to the DLPFC dataset
        save_path: Path to save cropped images

    Returns:
        adata: Prepared AnnData object
    """
    print("Loading data...")
    adata = read_10X_Visium(
        path=data_path,
        count_file='filtered_feature_bc_matrix.h5',
        load_images=True,
        quality='hires'
    )

    print(f"Loaded data: {adata.shape}")
    print(f"Number of spots: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")

    # Basic preprocessing
    print("Preprocessing gene expression data...")
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Crop images for each spot
    print("Cropping spot images...")
    adata = image_crop(
        adata,
        save_path=save_path,
        crop_size=50,
        target_size=224,
        verbose=False
    )

    return adata


def main():
    """Main execution function."""

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    data_path = '/scratch/harsha.vasamsetti/DeepST/data/DLPFC/151673'
    save_path = '/scratch/harsha.vasamsetti/testing/cropped_images'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    adata = load_and_prepare_data(data_path, save_path)

    # Image transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = SpatialDataset(adata, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # Initialize model
    n_genes = adata.n_vars
    print(f"\nInitializing model with {n_genes} genes...")

    model = MultiModalFusionModel(
        n_genes=n_genes,
        gene_emb_dim=128,
        image_emb_dim=128,
        fusion_dim=256,
        num_heads=8
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run inference to extract embeddings
    print("\nExtracting embeddings...")
    model.eval()

    all_fused_embeddings = []
    all_gene_embeddings = []
    all_image_embeddings = []
    all_attention_weights = []

    with torch.no_grad():
        for batch_idx, (gene_expr, images, indices) in enumerate(tqdm(dataloader)):
            gene_expr = gene_expr.to(device)
            images = images.to(device)

            # Forward pass
            fused_emb, gene_emb, img_emb, attn_weights = model(gene_expr, images)

            all_fused_embeddings.append(fused_emb.cpu().numpy())
            all_gene_embeddings.append(gene_emb.cpu().numpy())
            all_image_embeddings.append(img_emb.cpu().numpy())
            all_attention_weights.append(attn_weights.cpu().numpy())

    # Concatenate all embeddings
    fused_embeddings = np.vstack(all_fused_embeddings)
    gene_embeddings = np.vstack(all_gene_embeddings)
    image_embeddings = np.vstack(all_image_embeddings)

    print(f"\nEmbeddings extracted successfully!")
    print(f"Fused embeddings shape: {fused_embeddings.shape}")
    print(f"Gene embeddings shape: {gene_embeddings.shape}")
    print(f"Image embeddings shape: {image_embeddings.shape}")

    # Add embeddings to adata
    adata.obsm['gene_embeddings'] = gene_embeddings
    adata.obsm['image_embeddings'] = image_embeddings
    adata.obsm['fused_embeddings'] = fused_embeddings

    # Save results
    output_path = '/scratch/harsha.vasamsetti/testing/embeddings_output.h5ad'
    print(f"\nSaving results to {output_path}...")
    adata.write(output_path)

    # Print summary statistics
    print("\n" + "="*60)
    print("CROSS-ATTENTION FUSION COMPLETE")
    print("="*60)
    print(f"Gene embeddings mean: {gene_embeddings.mean():.4f}, std: {gene_embeddings.std():.4f}")
    print(f"Image embeddings mean: {image_embeddings.mean():.4f}, std: {image_embeddings.std():.4f}")
    print(f"Fused embeddings mean: {fused_embeddings.mean():.4f}, std: {fused_embeddings.std():.4f}")
    print("="*60)

    # Visualize some results
    print("\nComputing UMAP for visualization...")
    sc.pp.neighbors(adata, use_rep='fused_embeddings', n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)

    print(f"Found {len(adata.obs['leiden'].unique())} clusters")

    print("\nAll done! Results saved successfully.")

    return adata, model


if __name__ == "__main__":
    adata, model = main()