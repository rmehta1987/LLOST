import torch
from anCoder import anCoder
from feature_modules import FeatEncoder, FeatDecoder
from label_modules import LabelEncoder, LabelDecoder

class dualVAE(anCoder):
    def __init__(self, feat_arch, label_arch, feat_coupling_block, label_coupling_block, shared_coupling_block, 
                 shared_dim, dropout=0.5):
        super(dualVAE, self).__init__()
        
        self.Featenc = FeatEncoder(feat_arch,dropout)
        self.Featdec = FeatDecoder(feat_arch[::-1],dropout) # Decoder has symmetric layer architecture
        self.Labelenc = LabelEncoder(label_arch, dropout)
        self.Labeldec = LabelDecoder(label_arch[::-1],dropout) # Decoder has symmetric layer architecture
        self.shared_latent_size = shared_dim
        
        

        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        return mu, logvar 
    
    def encode(self, input):
       