import torch

from abc import ABC, abstractmethod
from einops import rearrange


class Quantizer(ABC, torch.nn.Module):
    '''
    Lightweight interface for quantization models
    '''
    def __init__(self,
                 model):
        super().__init__()
        self.quantizer = model

    @abstractmethod
    def quantize(self, x):
        pass

    @abstractmethod
    def dequantize(self, indices):
        pass

    @abstractmethod
    def get_codebook(self):
        pass

    def replace_codebook(self, _):
        pass


class FSQ(Quantizer):
    def __init__(self,
                 model):
        super().__init__(model)
        self.codebook_size = self.quantizer.codebook_size

    def quantize(self, x):
        tokens, indices = self.quantizer(x)
        return tokens, indices, 0
    
    def dequantize(self, indices):
        tokens = self.quantizer.indices_to_codes(indices) 
        return tokens
    
    def get_codebook(self):
        with torch.no_grad():
            codebook = self.quantizer.project_out(self.quantizer.implicit_codebook)
        return codebook
    

class VQ(Quantizer):
    def __init__(self,
                 model):
        super().__init__(model)
        self.codebook_size = self.quantizer.codebook_size

    def quantize(self, x):
        tokens, indices, commit_loss = self.quantizer(x)
        return tokens, indices, commit_loss
    
    def dequantize(self, indices):
        tokens = self.quantizer.project_out(self.quantizer.codebook[indices]) 
        return tokens
    
    def get_codebook(self):
        return self.quantizer.codebook
    

class NSVQ(Quantizer):
    def __init__(self,
                 model):
        super().__init__(model)
        self.codebook_size = self.quantizer.num_embeddings

    def quantize(self, x):
        if len(x.shape) == 2:
            tokens, indices, _ = self.quantizer(x)
            return tokens, indices, 0
        
        b, t, f = x.shape
        x = rearrange(x, 'b t f -> (b t) f')

        tokens, indices, _ = self.quantizer(x)

        tokens = rearrange(tokens, '(b t) f -> b t f', b=b, t=t)
        indices = rearrange(indices, '(b t) -> b t', b=b, t=t)

        return tokens, indices, 0
    
    def dequantize(self, indices):
        tokens = self.quantizer.project_out(self.quantizer.codebook[indices])
        return tokens
    
    def get_codebook(self):
        return self.quantizer.codebook.detach()
    
    def replace_codebook(self, replace_interval):
        self.quantizer.replace_unused_codebook(replace_interval)