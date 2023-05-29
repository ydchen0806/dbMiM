import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat,rearrange
from skimage.feature import hog
from vit_pytorch.vit import Transformer
from decision_net import DecisionNet

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64,
        hog = False,
        make_decision = False
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.HOG = hog
        self.make_decision = make_decision
        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.decision_net = DecisionNet(self.encoder.num_patches,self.encoder.image_patch_size,self.encoder.frame_patch_size)


    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        if self.make_decision:
            rand_indices = self.decision_net(patches).argsort(dim = -1)
        else:
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        if self.HOG:
            masked_patches = rearrange(masked_patches.cpu().numpy(), 'b p (h w d) -> b p h w d', h = self.encoder.image_patch_size, w = self.encoder.image_patch_size,d = self.encoder.frame_patch_size)
            for b in range(len(masked_patches)):
                for p in range(len(masked_patches[0])):
                    _, temp_hog = hog(masked_patches[b,p,:,:,2], orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True, multichannel=False)
                    masked_patches[b,p] = repeat(temp_hog, 'h w -> h w d', d = self.encoder.frame_patch_size)
            masked_patches = torch.tensor(rearrange(masked_patches, 'b p h w d -> b p (h w d)')).to(device)
        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)
        pred_pixel_values_sigmoid = torch.sigmoid(pred_pixel_values)
        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values_sigmoid, masked_patches)
        return recon_loss


if __name__ == '__main__':
    from vit_3d import ViT
    x = torch.randn(4,1,32,160,160).cuda()
    model = ViT(
                        image_size = 160,          # image size
                        frames = 32,               # number of frames
                        image_patch_size = 16,     # image patch size
                        frame_patch_size = 4,      # frame patch size
                        channels=1,
                        num_classes = 1000,
                        dim = 768 * 4,
                        depth = 12 * 2,
                        heads = 12 * 2,
                        mlp_dim = 5120,
                        dropout = 0.1,
                        emb_dropout = 0.1
                    )

    mae = MAE(
        encoder = model,
        masking_ratio = 0.85,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6,       # anywhere from 1 to 8
        hog = False
    ).cuda()
    print(mae(x))
    print('参数量M:', sum(p.numel() for p in mae.parameters()) / 1000000.0)
    # print(mae)
    # print(mae.encoder.transformer)
    torch.cuda.empty_cache()