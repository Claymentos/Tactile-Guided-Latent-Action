import torch
import h5py

from einops import rearrange

class AtariDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, indices):
        self.data_path = data_path
        self.indices = indices
        self.h5_file = None
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        self.open_h5_file()

        actual_idx = self.indices[idx]
        sample = self.h5_file[actual_idx][:]
        
        return sample
    
    def open_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')

    def __del__(self):
        """Ensure the file is closed when the dataset object is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()


class CoinRunDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path, 
                 window_size=16):
        
        self.data_path = data_path
        self.window_size = window_size

        with h5py.File(data_path, 'r') as f:
            self.level_size = f['0'].shape[0]
            self.total_len = len(f.keys()) * (self.level_size // self.window_size)

        self.h5_file = None
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        '''
        Each entry in the dataset consists of one level of the game, contatining 1000 steps (frames).
        '''
        self.open_h5_file()

        level_idx = idx // (self.level_size // self.window_size)
        actual_idx = idx - (level_idx * (self.level_size // self.window_size))

        sample = self.h5_file[str(level_idx)][actual_idx * self.window_size : (actual_idx + 1) * self.window_size]         # Shape (window_size, h, w, c)  
        sample = rearrange(torch.tensor(sample, dtype=torch.float32) / 255.0, "t h w c -> t c h w")
        
        return sample
    
    def open_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')

    def __del__(self):
        """Ensure the file is closed when the dataset object is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()





class CoinRunDataset2(torch.utils.data.Dataset):
    '''
    It can start at any given frame, adding a lot of redundancy to the dataset.
    '''
    def __init__(self, 
                 data_path, 
                 window_size=16):
        
        self.data_path = data_path
        self.window_size = window_size

        with h5py.File(data_path, 'r') as f:
            self.level_size = f['0'].shape[0]
            self.total_len = len(f.keys()) * (self.level_size - self.window_size + 1)

        self.h5_file = None
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        '''
        Each entry in the dataset consists of one level of the game, contatining 1000 steps (frames).
        '''
        self.open_h5_file()

        level_idx = idx // (self.level_size - self.window_size + 1)
        actual_idx = idx - (level_idx * (self.level_size - self.window_size + 1))

        sample = self.h5_file[str(level_idx)][actual_idx:actual_idx + self.window_size]         # Shape (window_size, h, w, c)  
        sample = rearrange(torch.tensor(sample, dtype=torch.float32) / 255.0, "t h w c -> t c h w")
        
        return sample
    
    def open_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')

    def __del__(self):
        """Ensure the file is closed when the dataset object is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()