import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
# %matplotlib inline

import torch
import torch.nn as nn
import numpy as np

# --- 1. Configuration ---
class Config:
    image_size = (60, 256, 256)  # Z, X, Y
    in_channels = 2              # C
    time_steps = 20              # T

    patch_depth = 10             # Example: divide Z=60 into 6 patches
    patch_height = 16            # Example: divide X=256 into 16 patches
    patch_width = 16             # Example: divide Y=256 into 16 patches

    embed_dim = 768              # Dimension for patch embeddings
    num_heads = 3               # Number of attention heads
    num_layers = 3              # Number of transformer encoder blocks
    mlp_ratio = 4.               # Ratio for feed-forward network expansion
    dropout_rate = 0.1

    # Calculate number of patches
    num_patches_z = image_size[0] // patch_depth
    num_patches_x = image_size[1] // patch_height
    num_patches_y = image_size[2] // patch_width
    num_spatial_patches_per_volume = num_patches_z * num_patches_x * num_patches_y
    total_patches = time_steps * num_spatial_patches_per_volume

    patch_dim = in_channels * patch_depth * patch_height * patch_width # Flattened patch size

    print(f"Total patches per 4D input: {total_patches}")
    print(f"Spatial Patches per volume per 4D input: {num_spatial_patches_per_volume}")
    print(f"Dimension of each flattened patch: {patch_dim}")

# --- 2. Patch Embedding Layer ---
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.patch_conv = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=(config.patch_depth, config.patch_height, config.patch_width),
            stride=(config.patch_depth, config.patch_height, config.patch_width)
        )
        # We'll use a linear projection instead of Conv3D if we want to strictly follow ViT's linear projection
        # For simplicity and efficiency with 3D patches, Conv3D is often used as a direct embedding
        # Alternative: Linear layer after manually extracting and flattening patches
        # self.linear_projection = nn.Linear(config.patch_dim, config.embed_dim)

    def forward(self, x): # x is (batch_size, time, channel, z, x, y)
        batch_size, T, C, Z, X, Y = x.shape
        
        # Reshape to (batch_size * time, channel, z, x, y) to apply 3D Conv
        x = x.view(batch_size * T, C, Z, X, Y)
        
        # Apply 3D convolution to get (batch_size * time, embed_dim, num_patches_z, num_patches_x, num_patches_y)
        x = self.patch_conv(x)
        
        # Flatten the spatial patch dimensions:
        # (batch_size * time, embed_dim, num_patches_z * num_patches_x * num_patches_y)
        x = x.flatten(2)
        
        # Transpose to get (batch_size * time, num_spatial_patches_per_volume, embed_dim)
        x = x.transpose(1, 2)
        print(x.shape, batch_size, T, C, Z, X, Y)
        # Reshape back to separate batch and time, then flatten time and spatial patches
        # (batch_size, T * num_spatial_patches_per_volume, embed_dim)
        x = x.reshape(batch_size, T * self.config.num_spatial_patches_per_volume, self.config.embed_dim)
        
        return x

# --- 3. Positional Encoding ---
class PositionalEncoding4D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Learnable positional embeddings for each possible patch position
        # We need to consider (time, z, x, y) for each patch
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.total_patches + 1, config.embed_dim) # +1 for CLS token
        )

    def forward(self, x): # x is (batch_size, num_tokens, embed_dim)
        # For simplicity, we assume the input already has a CLS token prepended
        # If not, add it here and adjust pos_embedding size
        
        # The actual assignment of specific pos_embedding vector to a (t,z,x,y) patch
        # needs to be handled during the data preparation or by carefully crafting this module.
        # For this conceptual code, we're using a single learnable embedding for all patches.
        # In a real implementation, you'd calculate which entry in pos_embedding corresponds
        # to the (t, z, x, y) index of each patch.
        
        return x + self.pos_embedding[:, :(x.size(1))] # Trim if x is shorter (e.g., during training)

# --- 4. Multi-Head Self-Attention (MHA) ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None): # x is (batch_size, num_tokens, embed_dim)
        print(x.shape)
        batch_size, num_tokens, embed_dim = x.shape

        # Linear projections for Q, K, V
        q = self.query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v are now (batch_size, num_heads, num_tokens, head_dim)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attention_scores is (batch_size, num_heads, num_tokens, num_tokens)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = torch.matmul(attention_weights, v)
        # context_vector is (batch_size, num_heads, num_tokens, head_dim)

        # Concatenate heads and project back
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)
        output = self.out_linear(context_vector)
        return output, attention_weights # Return weights for visualization/analysis

# --- 5. Feed-Forward Network (FFN) ---
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout_rate=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(), # Google's Error Linear Unit, common in Transformers
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

# --- 6. Transformer Encoder Block ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, mlp_ratio, dropout_rate)

    def forward(self, x, mask=None):
        # LayerNorm -> Attention -> Add&Norm
        attn_output, attn_weights = self.attn(self.norm1(x), mask=mask)
        x = x + attn_output # Residual connection

        # LayerNorm -> FFN -> Add&Norm
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output # Residual connection
        
        return x, attn_weights

# --- 7. Full 4D ViT Model ---
class FourDMedicalViT(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        self.config = config
        
        self.patch_embedding = PatchEmbedding(config)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        
        self.pos_embedding = PositionalEncoding4D(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        
        self.norm_final = nn.LayerNorm(config.embed_dim)

        # Optional: Head for classification/regression if num_classes is provided
        self.head = nn.Identity()
        if num_classes is not None:
            self.head = nn.Linear(config.embed_dim, num_classes)

    def forward(self, x, mask=None): # x is (batch_size, time, channel, z, x, y)
        batch_size = x.shape[0]
        
        # 1. Patch Embedding
        x = self.patch_embedding(x) # (batch_size, total_patches, embed_dim)
        
        # 2. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (batch_size, total_patches + 1, embed_dim)
        
        # 3. Add Positional Encoding
        x = self.pos_embedding(x)
        x = self.dropout(x)
        
        # 4. Transformer Encoder Blocks
        all_attention_weights = []
        for block in self.encoder_blocks:
            x, attn_weights = block(x, mask=mask)
            all_attention_weights.append(attn_weights)
            
        # 5. Final Normalization
        x = self.norm_final(x)
        
        # 6. Take CLS token for downstream tasks (e.g., classification)
        cls_output = x[:, 0] # (batch_size, embed_dim)
        
        output = self.head(cls_output)
        
        return output, all_attention_weights # Return attention weights for analysis


# --- Example Usage ---
if __name__ == "__main__":
    
    image = np.load("/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000049.npy")

    im1 = np.load("/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy")
    im2 = np.load("/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000069.npy")
    im1 = im1.astype(np.double)
    im2 = im2.astype(np.double)
    im = np.array([im1,im2]).astype(np.double)

    cfg = Config()
    torch.set_default_dtype(torch.float64)

    # Create a dummy 4D input
    # (batch_size, time, channel, z, x, y)
    # dummy_input = torch.randn(1, cfg.time_steps, cfg.in_channels, *cfg.image_size)

    dummy_input = torch.from_numpy(im).contiguous()
    print(f"Dummy input shape: {dummy_input.shape}")

    # Initialize the model
    # Let's say we want to classify this 4D image into 3 categories
    model = FourDMedicalViT(cfg, num_classes=3)
    print("Model initialized.")
    # print(model) # Uncomment to see the full model structure

    # Forward pass
    output, attention_weights_per_layer = model(dummy_input)
    print(f"Output shape (e.g., classification logits): {output.shape}")

    # Inspect attention weights (example from the last layer)
    if attention_weights_per_layer:
        last_layer_attn_weights = attention_weights_per_layer[-1]
        print(f"Last layer attention weights shape: {last_layer_attn_weights.shape}")
        # Shape is (batch_size, num_heads, num_tokens, num_tokens)
        # num_tokens includes the CLS token, so cfg.total_patches + 1

        # You can analyze these weights to understand how different patches (spatial and temporal)
        # attend to each other. For example, to see what a specific spatial patch at time T=0
        # attends to at time T=19.

    # Example for analyzing attention weights:
    # Let's say you want to see how the CLS token attends to other patches in the last layer
    if attention_weights_per_layer:
        cls_attn_to_patches = last_layer_attn_weights[0, :, 0, 1:] # Batch 0, all heads, CLS token's attention to other tokens
        print(f"\nCLS token attention to other patches (last layer, all heads): {cls_attn_to_patches.shape}")
        # Shape: (num_heads, total_patches)

        # To map back to original (t, z, x, y) positions:
        # Iterate through cfg.time_steps and cfg.num_spatial_patches_per_volume
        print("Example: Attention weights for CLS token from a single head (head 0):")
        head_0_attn = cls_attn_to_patches[0].cpu().numpy()
        
        # Reshape to (time_steps, num_spatial_patches_per_volume)
        reshaped_attn = head_0_attn.reshape(cfg.time_steps, cfg.num_spatial_patches_per_volume)
        
        # Now you can inspect 'reshaped_attn' to see attention values for each time step
        # and spatial patch within that time step.
        print(f"CLS attention to patches, reshaped (time, spatial_patches_per_volume): {reshaped_attn.shape}")
        print("First time step's spatial patch attention (first head, CLS token):")
        print(reshaped_attn[0, :5]) # First 5 spatial patches of the first time step
        print("Last time step's spatial patch attention (first head, CLS token):")
        print(reshaped_attn[-1, :5]) # First 5 spatial patches of the last time step

# %%



