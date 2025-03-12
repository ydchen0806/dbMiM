# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union, Optional
import torch.nn.functional as F
import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23')
from model_vit_3d import ViT
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    Enhanced with mask decision module support for adaptive masking
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        img_size: Tuple = (32, 160, 160),
        patch_size: Tuple = (4, 16, 16),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 8,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        kernel_size: Union[Sequence[int], int] = 3,
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.1,
        skip_connection: bool = False,
        show_feature: bool = True,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.img_size = img_size
        self.num_layers = 12
        self.show_feature = show_feature
        self.skip = skip_connection
        self.patch_size = patch_size
        self.feat_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.mha = nn.MultiheadAttention(16 ** 3, 4 ,batch_first=True)
        self.adapool = nn.AdaptiveAvgPool3d((16,16,16))
        self.vit = ViT(
                        image_size = img_size[1:],          # image size
                        frames = img_size[0],               # number of frames
                        image_patch_size = patch_size[1:],     # image patch size
                        frame_patch_size = patch_size[0],      # frame patch size
                        channels=1,
                        num_classes = 1000,
                        dim = hidden_size,
                        depth = 12, # 12
                        heads = num_heads,
                        mlp_dim = mlp_dim,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock( #(1,768,8,10,10)
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=kernel_size,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=kernel_size,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # skip=self.skip
        )
        self.dtrans = nn.Conv3d(feature_size * 2,feature_size * 2,kernel_size=(3, 3, 3), \
            stride=(int(self.patch_size[1]/self.patch_size[0]), 1, 1), padding=(1, 1, 1), bias=False)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        
        # HOG feature reconstruction head for multi-task pretraining
        self.hog_head = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, 9, kernel_size=1)  # 9 orientation bins for HOG
        )

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def get_patch_features(self, x_in):
        """Extract patch features for decision module to determine masking strategy"""
        # Get intermediate features from ViT
        _, hidden_states = self.vit.get_intermediate_features(x_in, return_patch_features=True)
        # Take features from an intermediate layer that has good semantic information
        patch_features = hidden_states[6]  # Using features from the middle layer
        return patch_features

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in, gt=None, mask_decisions=None, target_type='pixel'):
        """
        Forward pass with optional mask decisions from RL decision module
        
        Args:
            x_in: Input volume
            gt: Ground truth (optional)
            mask_decisions: Optional tensor with masking decisions (1: mask, 0: keep)
            target_type: 'pixel' for standard reconstruction, 'hog' for HOG feature reconstruction
            
        Returns:
            Depends on the mode:
            - For pretraining: loss if gt is provided, or reconstructed output
            - For inference: segmentation output
            - For feature extraction: intermediate features if show_feature is True
        """
        # Apply masking if mask_decisions is provided
        if mask_decisions is not None:
            # Get the transformer encoding with masking
            x, hidden_states_out = self.vit(x_in, mask_decisions=mask_decisions)
        else:
            # Standard forward pass without masking
            x, hidden_states_out = self.vit(x_in)
        
        # Feature extraction path
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        
        # Decoder path
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        if self.patch_size[0] != self.patch_size[1]:
            dec1 = self.dtrans(dec1)
        out = self.decoder2(dec1, enc1)
        
        # Different output heads based on target type
        if target_type == 'pixel':
            logits = self.out(out)
        elif target_type == 'hog':
            logits = self.hog_head(out)
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Return based on context
        if gt is None:
            # Inference or feature extraction mode
            if self.show_feature:
                return dec3, dec2, [dec3, dec2, dec1, out], torch.sigmoid(logits)
            else:
                return torch.sigmoid(logits)
        else:
            # Pretraining mode with loss calculation
            out = torch.sigmoid(logits)
            loss = F.mse_loss(out, gt)
            return loss
    
    def recons_visualize(self, x_in, gt_in, epoch, save_dir=None, mask_decisions=None):
        """Generate reconstructions for visualization and PSNR calculation"""
        with torch.no_grad():
            if mask_decisions is not None:
                # Get reconstruction with masking
                _, _, _, recons = self.forward(x_in, mask_decisions=mask_decisions)
            else:
                # Standard reconstruction
                _, _, _, recons = self.forward(x_in)
            
            # Calculate PSNR
            mse = F.mse_loss(recons, gt_in).item()
            if mse == 0:
                psnr = 100
            else:
                psnr = 10 * torch.log10(torch.tensor(1.0) / torch.tensor(mse))
            
            # Save visualization if directory is provided
            if save_dir is not None:
                import os
                import numpy as np
                from PIL import Image
                
                os.makedirs(save_dir, exist_ok=True)
                
                # Save input, masked input, and reconstruction
                input_np = x_in[0, 0].cpu().numpy()
                recons_np = recons[0, 0].cpu().numpy()
                gt_np = gt_in[0, 0].cpu().numpy()
                
                # Create a visualization with all images side by side
                # Take middle slice for 3D volumes
                if len(input_np.shape) == 3:
                    mid_slice = input_np.shape[0] // 2
                    input_slice = input_np[mid_slice]
                    recons_slice = recons_np[mid_slice]
                    gt_slice = gt_np[mid_slice]
                else:
                    input_slice = input_np
                    recons_slice = recons_np
                    gt_slice = gt_np
                
                # Normalize for visualization
                def normalize(img):
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    return (img * 255).astype(np.uint8)
                
                input_viz = normalize(input_slice)
                recons_viz = normalize(recons_slice)
                gt_viz = normalize(gt_slice)
                
                # Create masked input visualization if mask decisions provided
                if mask_decisions is not None:
                    masked_np = input_np.copy()
                    mask = mask_decisions[0].cpu().numpy()
                    
                    # Apply mask - set masked regions to a special value
                    if len(mask.shape) == 1:  # 1D mask for patches
                        # Need to expand mask to match input dimensions
                        # This is a simplified approach
                        mask_expanded = np.zeros_like(masked_np)
                        
                        # Map 1D mask to 3D volume
                        patch_size = self.patch_size
                        for i, m in enumerate(mask):
                            if m == 1:  # If patch is masked
                                # Calculate patch coordinates
                                z = (i // (self.feat_size[1] * self.feat_size[2])) * patch_size[0]
                                y = ((i % (self.feat_size[1] * self.feat_size[2])) // self.feat_size[2]) * patch_size[1]
                                x = ((i % (self.feat_size[1] * self.feat_size[2])) % self.feat_size[2]) * patch_size[2]
                                
                                # Set patch to masked value
                                z_end = min(z + patch_size[0], masked_np.shape[0])
                                y_end = min(y + patch_size[1], masked_np.shape[1])
                                x_end = min(x + patch_size[2], masked_np.shape[2])
                                
                                mask_expanded[z:z_end, y:y_end, x:x_end] = 1
                        
                        # Apply mask
                        masked_np[mask_expanded == 1] = 0.5  # Use gray for masked regions
                    
                    # Visualize masked input
                    if len(masked_np.shape) == 3:
                        masked_slice = masked_np[mid_slice]
                    else:
                        masked_slice = masked_np
                    
                    masked_viz = normalize(masked_slice)
                    
                    # Combine all images
                    combined = np.concatenate([input_viz, masked_viz, recons_viz, gt_viz], axis=1)
                else:
                    # Combine without masked input
                    combined = np.concatenate([input_viz, recons_viz, gt_viz], axis=1)
                
                # Save combined image
                Image.fromarray(combined).save(os.path.join(save_dir, f'recons_epoch_{epoch}.png'))
            
            return psnr.item()


# Add ViT extension to support masking
def extend_vit_with_masking():
    """Monkey patch the ViT class to support masking if necessary"""
    original_forward = ViT.forward
    
    def forward_with_masking(self, img, mask_decisions=None):
        """Extended forward pass with optional masking"""
        if mask_decisions is not None:
            # Process image through patch embedding
            x = self.patch_embedding(img)
            
            # Apply masking - replace masked patches with mask token or zero
            # We assume mask_decisions is of shape [B, num_patches] with binary values
            batch_size = x.shape[0]
            
            # Exclude CLS token for masking
            if hasattr(self, 'cls_token') and self.cls_token is not None:
                patches = x[:, 1:]
                
                # Expand mask to match patch dimensions
                expanded_mask = mask_decisions.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                
                # Apply mask: replace masked patches with zeros or learnable mask tokens
                # Here we use simple zero masking
                masked_patches = patches * (1 - expanded_mask.float())
                
                # Recombine with CLS token
                x = torch.cat([x[:, :1], masked_patches], dim=1)
            else:
                # No CLS token, apply masking directly
                expanded_mask = mask_decisions.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x = x * (1 - expanded_mask.float())
            
            # Continue with normal processing
            hidden_states = []
            for blk in self.blocks:
                x = blk(x)
                hidden_states.append(x)
            
            x = self.norm(x)
            
            # For pretraining tasks, typically return the hidden state of the last layer
            return x, hidden_states
        else:
            # Use original forward if no masking is needed
            return original_forward(self, img)
    
    # Add method to get intermediate features
    def get_intermediate_features(self, img, return_patch_features=False):
        """Extract intermediate features for decision module"""
        x = self.patch_embedding(img)
        
        hidden_states = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)
        
        x = self.norm(x)
        
        if return_patch_features:
            # Return features without CLS token if requested
            patch_features = [h[:, 1:] if h.shape[1] > 1 else h for h in hidden_states]
            return x, patch_features
        else:
            return x, hidden_states
    
    # Patch ViT with new methods if they don't exist
    if not hasattr(ViT, 'forward_with_masking'):
        ViT.forward = forward_with_masking
    
    if not hasattr(ViT, 'get_intermediate_features'):
        ViT.get_intermediate_features = get_intermediate_features


# Apply the extension
extend_vit_with_masking()


if __name__ == "__main__":
    import yaml
    from attrdict import AttrDict
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    cfg_file = 'pretraining_all.yaml'
    with open('/braindat/lab/chenyd/code/Miccai23/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    unetr = UNETR(
        in_channels=cfg.MODEL.input_nc,
        out_channels=cfg.MODEL.output_nc,
        img_size=cfg.MODEL.unetr_size,
        patch_size=cfg.MODEL.patch_size,
        feature_size=64,
        hidden_size=768,
        mlp_dim=2048,
        num_heads=8,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        kernel_size=cfg.MODEL.kernel_size,
        skip_connection=False,
        show_feature=True,
        dropout_rate=0.1).to(device)
    
    # Test parameter count
    print('Parameters (M): ', sum(param.numel() for param in unetr.parameters()) / 1e6)
    
    # Test with random input
    x = torch.randn(1, 1, 32, 160, 160).to(device)
    
    # Test with random mask decisions
    num_patches = (32 // 4) * (160 // 16) * (160 // 16)  # Z/patch_z * Y/patch_y * X/patch_x
    mask_decisions = torch.randint(0, 2, (1, num_patches)).to(device)
    
    # Test without masking
    _, _, feature, out = unetr(x)
    print("Output shape:", out.shape)
    for i, feat in enumerate(feature):
        print(f"Feature {i} shape:", feat.shape)
    
    # Test with masking
    print("\nTesting with mask decisions...")
    _, _, feature_masked, out_masked = unetr(x, mask_decisions=mask_decisions)
    print("Output shape with masking:", out_masked.shape)
    
    # Test patch feature extraction for decision module
    print("\nTesting patch feature extraction...")
    patch_features = unetr.get_patch_features(x)
    print("Patch features shape:", patch_features.shape)
    
    # Test HOG reconstruction
    print("\nTesting HOG reconstruction...")
    x_hog = torch.randn(1, 9, 32, 160, 160).to(device)  # Simulated HOG features
    hog_out = unetr(x, gt=x_hog, target_type='hog')
    print("HOG loss:", hog_out)
    
    torch.cuda.empty_cache()