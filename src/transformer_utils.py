import torch
import torch.nn as nn


def collate_fn(batch, config, mask_prob=0.05, mask_avg_len=1, test=False):
    """
    Collater for masking the acceleration data using 100.

    Arguments
    -----------------
    batch: list of tuples of PyTorch tensors corresponding to temporal and static batch tensors

    Returns
    -----------------
    X_batch: PyTorch tensor with shape (B * T * P)
    final_mask: PyTorch tensor with shape (B * T * P)
    y_batch: PyTorhc tensor with shape (B * Q)
    """

    # mask 15% of observations
    X_batch, y_batch = zip(*batch)
    X_batch = torch.stack(X_batch) 
    y_batch = torch.stack(y_batch) 

    # Create the initial mlm_mask
    mlm_mask = torch.rand(X_batch.shape[:-1]) < mask_prob * (X_batch[:,:,0] != config['pad_token_id'])

    # Create a Poisson distribution to draw from
    poisson = torch.distributions.Poisson(rate=mask_avg_len)  # Adjust rate as needed

    # Copy the initial mask
    final_mask = mlm_mask.clone()

    # Apply the Poisson distribution to extend the mask
    for i in range(X_batch.shape[0]):
        for j in range(X_batch.shape[1]):
            if mlm_mask[i, j]:
                # Draw from the Poisson distribution
                span = poisson.sample().int().item()
                end_idx = min(j + span, X_batch.shape[1])
                if config['pad_token_id'] not in X_batch[i, j:end_idx, :]:
                    final_mask[i, j:end_idx] = True

    final_mask = final_mask.unsqueeze(2)
    final_mask = final_mask.repeat(1, 1, X_batch.shape[-1]) 

    # if it is test set, mask the missing values for prediction.
    if test:
        final_mask[X_batch == config['pad_token_id']] = True

    return X_batch, final_mask, y_batch


class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, 'dim should be div by n_heads'
        self.head_dim = self.dim // self.n_heads
        self.in_proj = nn.Linear(dim, dim*3)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        b, t, c = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).permute(0,2,1,3) 
        k = k.view(b, t, self.n_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(b, t, self.n_heads, self.head_dim).permute(0,2,1,3)
        
        qkT = torch.matmul(q, k.transpose(-1,-2)) #* self.scale (B,n, T, d) * (B, n, d, T) -> ((B,n,T,T))
        if mask is not None:
            mask = mask.to(dtype=qkT.dtype, device=qkT.device)
            qkT = qkT.masked_fill(mask==0, float('-inf'))
        
        if mask is not None:
            mask = mask.to(dtype=qkT.dtype, device=qkT.device)
            qkT = qkT.masked_fill(mask==0, float('-inf'))

        qkT = torch.softmax(qkT * self.scale, dim=-1)
        qkT = self.attn_dropout(qkT)

        attn = torch.matmul(qkT, v).contiguous()
        attn = attn.permute(0,2,1,3).contiguous().view(b, t, c)
        out = self.out_proj(attn)
        
        return out

class FeedForward(nn.Module):
    def __init__(self,dim,dropout=0.1):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim)
        )
        
    def forward(self, x):
        return self.feed_forward(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0.1, mlp_dropout=0.):
        super().__init__()
        self.attn = MultiheadAttention(dim, n_heads, attn_dropout)
        self.ffd = FeedForward(dim, mlp_dropout)
        self.ln_1 = nn.LayerNorm(dim) 
        self.ln_2 = nn.LayerNorm(dim) 
        
    def forward(self, x, mask=None):
        x = x + self.attn(x, mask)
        x = self.ln_1(x)
        x = x + self.ffd(x)
        x = self.ln_2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, dim):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.class_embedding = nn.Linear(vocab_size, dim, bias=True)
        self.pos_embedding = nn.Embedding(max_len, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):        
        class_embed = self.class_embedding(x) 
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        pos_embed = self.pos_embedding(pos)
        embedding = class_embed + pos_embed
        return self.layer_norm(embedding)

class MLMBERT(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        
        # Initialize the embedding layer
        self.embedding = Embedding(vocab_size=config['vocab_size'], max_len=config['max_len'], dim=config['dim'])
        self.depth = config['depth']

        # Create a list of encoder blocks
        self.transformer_encoder = nn.ModuleList([
            EncoderBlock(
                dim=config['dim'],
                n_heads=config['n_heads'],
                attn_dropout=config['attn_dropout'],
                mlp_dropout=config['mlp_dropout']
            ) for _ in range(self.depth)
        ])
        
        # Initialize layer normalization layer
        # self.ln_f = nn.LayerNorm(config['dim'])
        
        self.pad_token_id = config['pad_token_id']
        self.mask_token_id = config['mask_token_id']

        # static features
        self.num_static_features = config['num_static_features']

        # This layer maps the hidden states to tri-axial acceleration
        self.decoder = nn.Linear(config['dim'], config['vocab_size'], bias=True)

        if self.num_static_features != 0:
            self.static_decoder = nn.Linear(self.num_static_features, config['vocab_size'], bias=True)

        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
        
    def create_src_mask(self, src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2) # N, 1, 1, src_len
    
    def forward(self, input_ids, static_ids=None):
        
        src_mask = self.create_src_mask(input_ids[:,:,0])
    
        enc_out = self.embedding(input_ids)
        
        for layer in self.transformer_encoder:
            enc_out = layer(enc_out, mask=src_mask)
        
        output = self.decoder(enc_out)

        if self.num_static_features != 0:
            assert self.static_ids is not None, "static features not given even though the model expects static features."
            static_ids =  static_ids.unsqueeze(1) # create an axis along the temporal dimension
            static_ids = static_ids.repeat(1, input_ids.shape[1], 1)
            static_out = self.static_decoder(static_ids)
            output += static_out

        mask_idx = (input_ids==self.mask_token_id)
        mask_preds = output[mask_idx]
        return {'mask_predictions':mask_preds, 'predictions': output}
