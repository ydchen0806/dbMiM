import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import h5py
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from glob import glob
from skimage.feature import hog
from skimage import exposure

class hdfDataset(Dataset):
    """HDF Dataset with optional HOG feature extraction"""

    def __init__(self, hdf_path, transform=None, if_hog=False, hog_params=None):
        """
        Args:
            hdf_path (string): Path to the hdf file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            if_hog (bool): Whether to return HOG features along with the image.
            hog_params (dict, optional): Parameters for HOG feature extraction.
        """
        self.hdf_path = hdf_path
        self.transform = transform
        self.if_hog = if_hog
        
        # Default HOG parameters
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': True,
            'multichannel': True
        }
        
        # Update with user provided parameters
        if hog_params is not None:
            self.hog_params.update(hog_params)

        self.data = h5py.File(self.hdf_path, 'r')
        self.length = len(self.data.keys())
        self.keys = list(self.data.keys())

    def __len__(self):
        return self.length

    def compute_hog(self, image):
        """Compute HOG features for the image"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for HOG computation
            if image.dim() == 3:  # [C, H, W]
                image_np = image.permute(1, 2, 0).numpy()
            else:  # [H, W]
                image_np = image.numpy()
        else:
            # Already numpy array
            image_np = image
            
        # Ensure image is normalized for HOG
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
            
        # For grayscale images
        if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
            self.hog_params['multichannel'] = False
            if len(image_np.shape) == 3:
                image_np = image_np.squeeze(-1)  # Remove channel dimension
        
        # Compute HOG features
        try:
            hog_features, hog_image = hog(
                image_np, 
                orientations=self.hog_params['orientations'],
                pixels_per_cell=self.hog_params['pixels_per_cell'],
                cells_per_block=self.hog_params['cells_per_block'],
                visualize=True,
                multichannel=self.hog_params['multichannel']
            )
            
            # Rescale HOG image for better visualization
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            # Convert to tensor
            if isinstance(image, torch.Tensor):
                hog_tensor = torch.from_numpy(hog_image_rescaled).float()
                # Add channel dimension if needed
                if hog_tensor.dim() == 2:
                    hog_tensor = hog_tensor.unsqueeze(0)
                return hog_tensor
            else:
                return hog_image_rescaled
                
        except Exception as e:
            print(f"Error computing HOG: {e}")
            # Return a blank tensor with same shape as input if HOG fails
            if isinstance(image, torch.Tensor):
                return torch.zeros_like(image)
            else:
                return np.zeros_like(image)

    def __getitem__(self, idx):
        image = self.data[self.keys[idx]][:]
        image = Image.fromarray(image)
        
        # Check if the image is grayscale (1 channel)
        if image.mode == 'L':
            # For HOG, it's often better to keep it grayscale
            if self.if_hog:
                pass  # Keep grayscale for HOG
            else:
                image = image.convert('RGB')  # Convert to RGB if HOG not needed
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # If HOG features requested
        if self.if_hog:
            hog_features = self.compute_hog(image)
            return image, image, hog_features  # Return (input, target, hog)
        else:
            return image, image  # Return (input, target)


class ImageDataset(Dataset):
    """Dataset for loading images from directory with optional HOG feature extraction"""
    
    def __init__(self, img_dir, transform=None, if_hog=False, hog_params=None):
        """
        Args:
            img_dir (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
            if_hog (bool): Whether to return HOG features along with the image.
            hog_params (dict, optional): Parameters for HOG feature extraction.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.if_hog = if_hog
        self.img_list = sorted(glob(os.path.join(self.img_dir, '*.png')))
        
        # Default HOG parameters
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': True,
            'multichannel': True
        }
        
        # Update with user provided parameters
        if hog_params is not None:
            self.hog_params.update(hog_params)

    def __len__(self):
        return len(self.img_list)
    
    def compute_hog(self, image):
        """Compute HOG features for the image"""
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy for HOG computation
            if image.dim() == 3:  # [C, H, W]
                image_np = image.permute(1, 2, 0).numpy()
            else:  # [H, W]
                image_np = image.numpy()
        else:
            # Already numpy array
            image_np = image
            
        # Ensure image is normalized for HOG
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
            
        # For grayscale images
        if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
            self.hog_params['multichannel'] = False
            if len(image_np.shape) == 3:
                image_np = image_np.squeeze(-1)  # Remove channel dimension
        
        # Compute HOG features
        try:
            hog_features, hog_image = hog(
                image_np, 
                orientations=self.hog_params['orientations'],
                pixels_per_cell=self.hog_params['pixels_per_cell'],
                cells_per_block=self.hog_params['cells_per_block'],
                visualize=True,
                multichannel=self.hog_params['multichannel']
            )
            
            # Rescale HOG image for better visualization
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            
            # Convert to tensor
            if isinstance(image, torch.Tensor):
                hog_tensor = torch.from_numpy(hog_image_rescaled).float()
                # Add channel dimension if needed
                if hog_tensor.dim() == 2:
                    hog_tensor = hog_tensor.unsqueeze(0)
                return hog_tensor
            else:
                return hog_image_rescaled
                
        except Exception as e:
            print(f"Error computing HOG: {e}")
            # Return a blank tensor with same shape as input if HOG fails
            if isinstance(image, torch.Tensor):
                return torch.zeros_like(image)
            else:
                return np.zeros_like(image)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = np.array(img)
            img = torch.from_numpy(img).float()
            # Check if the tensor needs to be rearranged
            if img.dim() == 3 and img.shape[-1] in (1, 3):  # HWC format
                img = img.permute(2, 0, 1)  # Convert to CHW format
        
        # If HOG features requested
        if self.if_hog:
            hog_features = self.compute_hog(img)
            return img, img, hog_features  # Return (input, target, hog)
        else:
            return img, img  # Return (input, target)


class EM3DDataset(Dataset):
    """Dataset for 3D EM data with optional HOG feature extraction"""
    
    def __init__(self, data_path, transform=None, if_hog=False, hog_params=None, patch_size=(32, 128, 128)):
        """
        Args:
            data_path (string): Path to 3D data file or directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            if_hog (bool): Whether to return HOG features along with the image.
            hog_params (dict, optional): Parameters for HOG feature extraction.
            patch_size (tuple): Size of patches to extract (D, H, W).
        """
        self.data_path = data_path
        self.transform = transform
        self.if_hog = if_hog
        self.patch_size = patch_size
        
        # Load data based on file type
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            self.data = h5py.File(data_path, 'r')
            self.volume = self.data['volume'][:]  # Adjust key based on your HDF5 structure
        elif data_path.endswith('.npy'):
            self.volume = np.load(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Calculate number of patches
        self.depth, self.height, self.width = self.volume.shape
        self.num_d_patches = max(1, (self.depth - patch_size[0]) // (patch_size[0] // 2) + 1)
        self.num_h_patches = max(1, (self.height - patch_size[1]) // (patch_size[1] // 2) + 1)
        self.num_w_patches = max(1, (self.width - patch_size[2]) // (patch_size[2] // 2) + 1)
        self.total_patches = self.num_d_patches * self.num_h_patches * self.num_w_patches
        
        # Default HOG parameters for 3D data
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': True,
            'multichannel': False
        }
        
        # Update with user provided parameters
        if hog_params is not None:
            self.hog_params.update(hog_params)
    
    def __len__(self):
        return self.total_patches
    
    def compute_3d_hog(self, volume):
        """Compute HOG features for each slice of the 3D volume"""
        D, H, W = volume.shape
        hog_volume = np.zeros((D, H, W), dtype=np.float32)
        
        # Process each slice separately
        for d in range(D):
            slice_2d = volume[d]
            
            # Normalize slice for HOG
            if slice_2d.max() > 1.0:
                slice_2d = slice_2d / 255.0
            
            # Compute HOG for the 2D slice
            try:
                _, hog_image = hog(
                    slice_2d, 
                    orientations=self.hog_params['orientations'],
                    pixels_per_cell=self.hog_params['pixels_per_cell'],
                    cells_per_block=self.hog_params['cells_per_block'],
                    visualize=True,
                    multichannel=False
                )
                
                # Rescale HOG image
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                hog_volume[d] = hog_image_rescaled
                
            except Exception as e:
                print(f"Error computing HOG for slice {d}: {e}")
                hog_volume[d] = np.zeros_like(slice_2d)
        
        # Convert to tensor
        hog_tensor = torch.from_numpy(hog_volume).float()
        
        # Add channel dimension
        hog_tensor = hog_tensor.unsqueeze(0)
        
        return hog_tensor
    
    def __getitem__(self, idx):
        # Calculate patch coordinates
        d_idx = idx // (self.num_h_patches * self.num_w_patches)
        h_idx = (idx % (self.num_h_patches * self.num_w_patches)) // self.num_w_patches
        w_idx = (idx % (self.num_h_patches * self.num_w_patches)) % self.num_w_patches
        
        d_start = min(d_idx * (self.patch_size[0] // 2), self.depth - self.patch_size[0])
        h_start = min(h_idx * (self.patch_size[1] // 2), self.height - self.patch_size[1])
        w_start = min(w_idx * (self.patch_size[2] // 2), self.width - self.patch_size[2])
        
        # Extract patch
        patch = self.volume[
            d_start:d_start + self.patch_size[0],
            h_start:h_start + self.patch_size[1],
            w_start:w_start + self.patch_size[2]
        ]
        
        # Apply transform if provided
        if self.transform:
            patch = self.transform(patch)
        
        # Convert to tensor if not already
        if not isinstance(patch, torch.Tensor):
            patch = torch.from_numpy(patch).float()
        
        # Add channel dimension if needed
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        
        # If HOG features requested
        if self.if_hog:
            hog_features = self.compute_3d_hog(patch.squeeze(0).numpy())
            return patch, patch, hog_features  # Return (input, target, hog)
        else:
            return patch, patch  # Return (input, target)


if __name__ == '__main__':
    hdf_path = '/h3cstore_ns/screen_generate/test/kodak'
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),
            transforms.ToTensor()])
    
    # Test ImageDataset with HOG
    print("Testing ImageDataset with HOG...")
    dataset = ImageDataset(hdf_path, transform=transform_train, if_hog=True)
    save_dir = '/data/ydchen/VLP/mage-main/temp_img'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Dataset length: {len(dataset)}")
    
    for i in range(5):
        img, _, hog_features = dataset[i]
        print(f"Image shape: {img.shape}, HOG shape: {hog_features.shape}")
        
        # Visualize image and HOG
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(hog_features.squeeze(0), cmap='gray')
        plt.title('HOG Features')
        
        plt.savefig(os.path.join(save_dir, f'hog_sample_{i}.png'), bbox_inches='tight', pad_inches=0, dpi=360)
        plt.close()
    
    # Test with 3D data if available
    try:
        print("\nTesting EM3DDataset...")
        # Simulate a small 3D volume for testing
        test_volume = np.random.rand(64, 256, 256)
        np.save('/tmp/test_volume.npy', test_volume)
        
        em_dataset = EM3DDataset('/tmp/test_volume.npy', if_hog=True, patch_size=(32, 128, 128))
        print(f"EM dataset length: {len(em_dataset)}")
        
        patch, _, hog = em_dataset[0]
        print(f"Patch shape: {patch.shape}, HOG shape: {hog.shape}")
        
        # Visualize middle slice
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(patch[0, patch.shape[1]//2], cmap='gray')
        plt.title('Middle Slice')
        
        plt.subplot(1, 2, 2)
        plt.imshow(hog[0, hog.shape[1]//2], cmap='gray')
        plt.title('HOG Middle Slice')
        
        plt.savefig(os.path.join(save_dir, 'em3d_hog_sample.png'), bbox_inches='tight', pad_inches=0, dpi=360)
        plt.close()
        
    except Exception as e:
        print(f"Skipping EM3D test: {e}")