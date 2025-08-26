import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        
        B, C, H, W = x.shape
        device = x.device
        yy = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        xx = torch.linspace(0, 1, W, device=device).unsqueeze(0).repeat(H, 1)
        pos = torch.stack([xx, yy], dim=0)        
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1) 
        pos = pos.flatten(2).permute(0,2,1)       
        
        repeat_factor = max(1, self.num_pos_feats // 2)
        pos = pos.repeat(1,1,repeat_factor)       
        return pos

class QueryDetector(nn.Module):
    def __init__(self, num_classes, num_queries=12, hidden_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        
        backbone = resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  
        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=hidden_dim)

        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        B = x.shape[0]
        feats = self.backbone(x)                   
        feats = self.proj(feats)                   
        B, C, Hf, Wf = feats.shape
        src = feats.flatten(2).permute(2,0,1)      
        pos = self.pos_enc(feats)                  
        pos = pos.permute(1,0,2)                   
        memory = src + pos

        
        q = self.query_embed.weight.unsqueeze(1).repeat(1,B,1)  

        
        decoded = self.decoder(tgt=q, memory=memory)            
        decoded = decoded.permute(1,0,2)                        

        logits = self.class_head(decoded)                       
        boxes  = self.bbox_head(decoded)                        
        return {"pred_logits": logits, "pred_boxes": boxes}
