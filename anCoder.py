import torch.nn as nn


class anCoder(nn.Module):
    '''
        Abstract class for defining an encoder and decoder for a VAE architecture
        The class that inherits must implement the forward method and weight initalization for 
        the neural networks
    '''
    @abstractmethod
    def __init__(self, arch, dropout=0.5):
        self.dims = arch
        self.dropout = dropout
        super(anCoder).__init__()
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def init_weights(self):
        raise NotImplementedError