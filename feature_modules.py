import torch
from anCoder import anCoder

class FeatEncoder(anCoder):
    def __init__(self, arch, dropout=0.5):
        super(FeatEncoder, self).__init__()
        self.q_dims = arch # inference network
        self.dropout = dropout
        
        # Last dimension of the inference network, q, is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        return mu, logvar 
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def init_weights(self):
        ''' 
            Initalize weights using Xavier initalization 
            Intalize biases using Normal initaliziation
        '''
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
class FeatDecoder(anCoder):
    def __init__(self, arch, dropout=0.5):
        super(FeatDecoder, self).__init__()
        self.p_dims = arch # generative network
        self.dropout = dropout
        
        # Last dimension of the generative network, p, is the reconstructed original
        temp_p_dims = self.p_dims[:-1] + [self.p_dims[-1] * 2]
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_p_dims[:-1], temp_p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        reconstructed = self.decode(input)
        return reconstructed
    
    def decode(self, input):
        '''
            Decodes the inference network parameters to the distribution of 
            the generative network
        '''
        h = input
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
            
            
        r = torch.exp(h[:, :self.p_dims[-1]])
        return r
    
    def init_weights(self):
        ''' 
            Initalize weights using Xavier initalization 
            Intalize biases using Normal initaliziation
        '''
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

