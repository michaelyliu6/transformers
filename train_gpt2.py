print("Starting...")

import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch
import math

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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

         # flash attention
         # processes the entire attention calculation into smaller chunks 
         # memory efficient way of calculating attention without storing full attention matrix 
         # recomputes certain values as needed
         # better utilizes accesses to GPU memory bandwidth 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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

        # weight tie the embedding and unembedding matrix
        self.transformer.wte.weight = self.lm_head.weight

        # init params by applying _init_weights to all submodules 
        self.apply(self.init_weights)

    # this is how gpt2 was initalized in https://github.com/openai/gpt-2/blob/master/src/model.py#L152-L167, but didn't find significant improvement from this
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config['n_layer']) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # idx is shape [batch_size, block_size] (still in token form, not embedding)
        B, T = idx.shape
        
        positions = torch.arange(0, T, dtype=torch.long, device=idx.device) # Simple tensor of positions [0, 1, 2, 3, 4, ..., T - 1]
        positional_embeddings = self.transformer.wpe(positions)

        token_embeddings = self.transformer.wte(idx)

        x = positional_embeddings + token_embeddings # input to the actual transformer blocks 

        for block in self.transformer.h: # feeding the input through the transformer blocks 
            x = block(x)

        x = self.transformer['ln_f'](x) # final layer form

        logits = self.lm_head(x) # final linear classifier 

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

        




model_args = {
    'vocab_size': 50304,
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

# with torch.no_grad(): # explicitly tell pytorch not to keep track of gradients to save some performance 
#     for k in hf_sd_keys:
#         if any(k.endswith(w) for w in transposed_keys): # for keys related to 
#             sd[k].copy_(hf_model_state_dict[k].t())
#         else:
#             sd[k].copy_(hf_model_state_dict[k])


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
#                                                      [block_size, n_embd]
#                                                               |
#                                                               V
# *--- Blocks -----------------------------------------------------------------------------------------------------------
#                                     element-wise addition with attention(layer_norm(input)) [block_size, n_embd]
#                                                               |
#                                                               V
#                                      element-wise addition with mlp(layer_norm(input)) [block_size, n_embd]
# *--- Blocks -----------------------------------------------------------------------------------------------------------
#                                                               |
#                                                       Layer Normalization
#                                                               |
#                                                               V
#                                               multiply by linear classifier [n_embd, vocab_size]
#                                                               |
#                                                               V
#                                                    [block_size, vocab_size]

# -----------------------------------------------------------------------------
import tiktoken
import time

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")



train_loader = DataLoaderLite(B=16, T=1024) # generated x and y of shape [16, 1024] 



# used for unlocking better performance by better utilizing the GPU
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('high')


scratch_model.eval()
scratch_model.to(device)

# allow pytorch to automatically read through the model and make optimizations 
# https://pytorch.org/docs/stable/generated/torch.compile.html
scratch_model = torch.compile(model)

# Initialize the AdamW optimizer with a learning rate of 3e-4
optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=3e-4)

# Start a training loop that will run for 50 iterations
for i in range(50):
    # fetch next batch of training data from DataLoaderLite
    x, y = train_loader.next_batch()

    # push training data to device
    x, y = x.to(device), y.to(device)


    # Zero out the gradients from the previous iteration
    optimizer.zero_grad()
    
    # Forward pass: compute the model's output (logits) and loss
    # let pytorch selectively downcasting the precision of some operations to speed up training while maintaining accuracy and reducing memory usage 
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits, loss = model(x, y)
    
    # Backward pass: compute gradients of the loss with respect to model parameters
    loss.backward()
    
    # Update the model's parameters using the computed gradients
    optimizer.step()
    
    # wait for the GPU to finish work
    torch.cuda.synchronize()

     # calculate time difference in miliseconds
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")


# import tiktoken

# num_return_sequences = 5
# max_sequence_length = 30

# encoder = tiktoken.get_encoding('gpt2')
# tokens = encoder.encode("Hello, I am an language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0) # (1, 8)
# tokens = tokens.repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)




# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# with torch.no_grad():
#     while x.size(1) < max_sequence_length: # keep generating next token until max_sequence_length is reached
#         logits = scratch_model(x)
#         # [batch_size, block_size, vocab_size] -> [batch_size, block_size, vocab_size] 

#         logits = logits[:,-1,:]
#         # [batch_size, vocab_size] selects the logits for the last token in each sequence 

#         probs = nn.functional.softmax(logits, dim=-1) 
#         # [batch_size, vocab_size] applies a softmax function to convert logits to probabiltiies 

#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) 
#         # topk_probs shape: [batch_size, 50]
#         # topk_indices shape: [batch_size, 50] 

#         ix = torch.multinomial(topk_probs, 1) 
#         # [B, 1] Performs a weighted random sampling from the top-k probabilities.

#         xcol = torch.gather(topk_indices, -1, ix)
#         # [B, 1] maps the sampled index back to the original vocabulary index.

#         x = torch.cat((x, xcol), dim=1)
#         # [B, L+ 1] appends the newly generated token to the end of the sequence 

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_sequence_length].tolist()
#     decoded = encoder.decode(tokens)
#     print(">", decoded) 


print("Finished!!")
