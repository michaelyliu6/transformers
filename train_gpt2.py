print("Starting...")

import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch
import math

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connections 
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config['block_size'], config['block_size']))
                                     .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        # Input: x (shape: [batch_size, seq_length, n_embd])
        # Output: x (shape: [batch_size, seq_length, n_embd])

        super().__init__()
        # y = xA^T + b
        # Fully-connected/Feed-forward layer
        # Expands the model to higher dimensional space to capture more complex representations
        self.c_fc    = nn.Linear(config['n_embd'], 4 * config['n_embd']) 

        # Activation Function https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
        # Similar to Relu (x > 0 ? x : 0), but allows for signfiicantly smoother gradient transitions
        self.gelu    = nn.GELU()

        # Projection layer 
        # Converting the back to originial dimensionality 
        self.c_proj  = nn.Linear(4 * config['n_embd'], config['n_embd'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config['vocab_size'], config['n_embd']), # Token embedding 
            'wpe': nn.Embedding(config['block_size'], config['n_embd']), # Positional embedding
            'h': nn.ModuleList([Block(config) for _ in range(config['n_layer'])]), # hidden layers (attention/mlp/normalization, etc)
            'ln_f': nn.LayerNorm(config['n_embd']),
        })

        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, idx): # idx is shape [batch_size, block_size] (still in token form, not embedding)
        B, T = idx.shape
        
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device) # Simple tensor of positions [0, 1, 2, 3, 4, ..., T - 1]
        positional_embeddings = self.transformer.wpe(positions)

        token_embeddings = self.transformer.wte(idx)

        x = positional_embeddings + token_embeddings # input to the actual transformer blocks 

        for block in self.transformer.h: # feeding the input through the transformer blocks 
            x = block(x)

        x = self.transformer['ln_f'](x) # final layer form

        logits = self.lm_head(x) # final linear classifier 

        return logits

        




model_args = {
    'vocab_size': 50257,
    'block_size': 1024,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
} # for the 124M parameter GPT model 

scratch_model = GPT(model_args)
sd = scratch_model.state_dict()


sd_keys = sd.keys()

# print(sd_keys)

sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # ??????


hugging_face_model = GPT2LMHeadModel.from_pretrained('gpt2')
hf_model_state_dict = hugging_face_model.state_dict()


hf_sd_keys = hf_model_state_dict.keys()

print("\n")

# print(hf_sd_keys)

hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.bias')] # ??? 
hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.masked_bias')] # ????


# In the OpenAI model from HuggingFace, they use a "Conv1D" module, but we are just going to use a "Linear module"
# So we just need to transpose the weights when we import them into our model skeleton 
transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

with torch.no_grad(): # explicitly tell pytorch not to keep track of gradients to save some performance 
    for k in hf_sd_keys:
        if any(k.endswith(w) for w in transposed_keys): # for keys related to 
            sd[k].copy_(hf_model_state_dict[k].t())
        else:
            sd[k].copy_(hf_model_state_dict[k])


# print(sd_keys)




# Input:                                            [block_size, vocab_size]
#                                                               |
#  wte                                  multiply by the Token Embedding Layer ([vocab_size, n_embd]) 
#                                                               |
#                                                               V
#                                                   [block_size, n_embd]
#                                                               |
#  wpe                         element-wise addition with the Positional Embedding Layer ([block_size, n_embd])
#                                                               |
#                                                               V
#                                                   [block_size, n_embd]
#                                                               |
#                                                               V
# *--- Blocks -----------------------------------------------------------------------------------------------------------
#                                                     Normalization Layer
#                                                               | 
#                                                               V
#                                                    [block_size, n_embd]
#                                                               | 
#                                             
# *--- Blocks -----------------------------------------------------------------------------------------------------------

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


import tiktoken

num_return_sequences = 5
max_sequence_length = 30

encoder = tiktoken.get_encoding('gpt2')
tokens = encoder.encode("Hello, I am an language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0) # (1, 8)
tokens = tokens.repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

scratch_model.eval()
scratch_model.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

with torch.no_grad():
    while x.size(1) < max_sequence_length: # keep generating next token until max_sequence_length is reached
        logits = scratch_model(x)
        # [batch_size, block_size, vocab_size] -> [batch_size, block_size, vocab_size] 

        logits = logits[:,-1,:]
        # [batch_size, vocab_size] selects the logits for the last token in each sequence 

        probs = nn.functional.softmax(logits, dim=-1) 
        # [batch_size, vocab_size] applies a softmax function to convert logits to probabiltiies 

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) 
        # topk_probs shape: [batch_size, 50]
        # topk_indices shape: [batch_size, 50] 

        ix = torch.multinomial(topk_probs, 1) 
        # [B, 1] Performs a weighted random sampling from the top-k probabilities.

        xcol = torch.gather(topk_indices, -1, ix)
        # [B, 1] maps the sampled index back to the original vocabulary index.

        x = torch.cat((x, xcol), dim=1)
        # [B, L+ 1] appends the newly generated token to the end of the sequence 

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_sequence_length].tolist()
    decoded = encoder.decode(tokens)
    print(">", decoded) 


print("Finished!!")
