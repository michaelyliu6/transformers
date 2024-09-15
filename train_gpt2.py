import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch
import math
import inspect
import os 
import time

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
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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



    # About AdamW:
    # Basic Gradient Descent:
    #
    # Uses only the current gradient to update weights.
    # Weight update: new_weight = old_weight - learning_rate * current_gradient
    #
    #
    # Adam and similar optimizers:
    #
    # Still use the current gradient, but in a more sophisticated way.
    # They don't directly use past gradients, but they keep track of gradient statistics.
    #
    #
    # What Adam actually does:
    #
    # Calculates the current gradient at each step, just like basic gradient descent.
    # Maintains running averages of gradient statistics (not past gradients themselves).
    # Uses these statistics to adjust how the current gradient is applied.
    #
    #
    # The role of decay rate:
    #
    # Controls how these running averages are updated with each new gradient.
    # It doesn't apply to past gradients directly, but to these statistical summaries.
    #
    #
    # An analogy:
    #
    # Imagine you're steering a ship. Basic gradient descent is like turning the wheel based solely on your current position.
    # Adam is like considering your current position, but also factoring in your recent trajectory and speed of turning.
    #
    # weight decay: it's equivalent to adding a regularization term to the loss function: L2 regularization.
    #   Encourages smaller weights, potentially leading to simpler models.
    # lr: learning rate 
    # optim_groups: parameters that are or aren't weight decayed
    # betas: β₁, β₂: decay rates for moment estimates
    #   Control the balance between using recent and historical gradient information.
    # eps: parameter in the optimizer configuration refers to a small constant value added for numerical stability instead of 0 for divions        
    # fused: boolean: use fused kernals in CUDA if available
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer




model_args = {
    'vocab_size': 50304,
    'block_size': 1024,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
} # for the 124M parameter GPT model 

# get model
model = GPT(model_args)
sd = model.state_dict()


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
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# https://pytorch.org/docs/stable/notes/ddp.html
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    # initializes the distributed environment. Each process gets a unique rank and is assigned to a specific GPU.
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)


# used for unlocking better performance by better utilizing the GPU
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('high')


model.eval()
model.to(device)

# allow pytorch to automatically read through the model and make optimizations 
# https://pytorch.org/docs/stable/generated/torch.compile.html
model = torch.compile(model)
if ddp:
    # wrap model in DDP which handles the distribution of the model across GPUs.
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model




max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Start a training loop that will run for 50 iterations
for step in range(max_steps):

    # Zero out the gradients from the previous iteration
    optimizer.zero_grad()
    
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        t0 = time.time()

        # fetch next batch of training data from DataLoaderLite
        x, y = train_loader.next_batch()

        # push training data to device
        x, y = x.to(device), y.to(device)

        # Forward pass: compute the model's output (logits) and loss
         # let pytorch selectively downcasting the precision of some operations to speed up training while maintaining accuracy and reducing memory usage 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
    
        # only keep track of the scalar loss without the gradient graph
        loss_accum += loss.detach()

        if ddp:
            # controls when gradients are synchronized across GPUs. It's set to True only on the last micro-step of each iteration to allow for gradient accumulation.
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss.backward()

    if ddp:
        # average the loss across each GPU after each iteration
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Gradient clipping: If the norm exceeds the specified maximum (1.0 in this case), it scales down all gradients proportionally so that their norm equals the maximum.
    # Stablize training by prevent gradients from becoming too large (exploding gradients) 
    # norm is the total of the norms of all the gradients before clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Update the model's parameters using the computed gradients
    optimizer.step()
    
    # wait for the GPU to finish work
    torch.cuda.synchronize()

     # calculate time difference in miliseconds
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

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
#         logits = model(x)
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

