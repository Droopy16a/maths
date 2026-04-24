import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Le tutoriel utilise DenseNet121
        densenet = models.densenet121(weights='DEFAULT')
        self.features = densenet.features
        
        # On remplace la couche de classification finale par une convolution 1x1 
        # pour réduire la dimensionnalité à d_model (ex: 256)
        self.conv = nn.Conv2d(1024, d_model, kernel_size=1)
        
    def forward(self, x):
        features = self.features(x) # [Batch, 1024, H, W]
        features = self.conv(features) # [Batch, d_model, H, W]
        
        # Aplatissement pour le Transformer: [Batch, Sequence_Length, d_model]
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1) 
        return features

class HMERModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.encoder = ImageEncoder(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding 1D pour le texte
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) 
        
        # Le Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Couche finale (fc_out) pour mapper à la taille du vocabulaire
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward_encoder(self, images):
        return self.encoder(images)