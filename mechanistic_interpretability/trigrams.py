#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/michaelyliu6/transformers/blob/main/ARENA_November_2024_Challenge_Trigrams.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Monthly Algorithmic Challenge (November 2024): Trigrams (RUN IN COLAB ^^)
# 
# This post is the seventh from a sequence of monthly mechanistic interpretability challenges. They are designed in the spirit of [Stephen Casper's challenges](https://www.lesswrong.com/posts/KSHqLzQscwJnv44T8/eis-vii-a-challenge-for-mechanists), but with the more specific aim of working well in the context of the rest of the ARENA material, and helping people put into practice all the things they've learned so far.
# 
# If you prefer, you can access the Streamlit page [here](https://arena3-chapter1-transformer-interp.streamlit.app/Monthly_Algorithmic_Problems).
# 
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/trigrams.png" width="350">

# ## Setup

# In[ ]:


try:
    import google.colab # type: ignore
    IN_COLAB = True
except:
    IN_COLAB = False

import os, sys
chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"

if IN_COLAB:
    # Install packages
    get_ipython().run_line_magic('pip', 'install transformer_lens')
    get_ipython().run_line_magic('pip', 'install einops')
    get_ipython().run_line_magic('pip', 'install jaxtyping')
    get_ipython().run_line_magic('pip', 'install git+https://github.com/callummcdougall/eindex.git')
    get_ipython().run_line_magic('pip', 'install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python')

    # Code to download the necessary files (e.g. solutions, test funcs)
    if not os.path.exists(chapter):
        get_ipython().system('wget https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/main.zip')
        get_ipython().system("unzip /content/main.zip 'ARENA_3.0-main/chapter1_transformer_interp/exercises/*'")
        sys.path.append(f"/content/{repo}-main/{chapter}/exercises")
        os.remove("/content/main.zip")
        os.rename(f"{repo}-main/{chapter}", chapter)
        os.rmdir(f"{repo}-main")
        os.chdir(f"{chapter}/exercises")
else:
    chapter_dir = r"./" if chapter in os.listdir() else os.getcwd().split(chapter)[0]
    sys.path.append(chapter_dir + f"{chapter}/exercises")


# In[ ]:


import os
import sys
from pathlib import Path

import numpy as np
import torch as t
from eindex import eindex
from transformer_lens import utils, HookedTransformer, ActivationCache
from jaxtyping import Bool, Float, Int
from typing import List, Set, Tuple
import einops

import transformer_lens.utilities.addmm as aaddmm
import transformer_lens.utils as utils
import torch.nn.functional as F

t.set_grad_enabled(False)

exercises_dir = Path.cwd()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "november24_trigrams"
assert section_dir.name == "november24_trigrams", "Please navigate to the correct directory using os.chdir"

from monthly_algorithmic_problems.november24_trigrams.dataset import BigramDataset
from monthly_algorithmic_problems.november24_trigrams.model import create_model
from plotly_utils import imshow

device = t.device("cpu")


# ## Prerequisites
# 
# The following ARENA material should be considered essential:
# 
# * **[1.1] Transformer from scratch** (sections 1-3)
# * **[1.2] Intro to Mech Interp** (sections 1-3)
# 
# The following material isn't essential, but is recommended:
# 
# * **[1.2] Intro to Mech Interp** (section 4)
# * **[1.7] Balanced Bracket Classifier** (all sections)
# * Previous algorithmic problems in the sequence
# 

# ## Difficulty
# 
# I estimate that this problem is slightly easier than average problem in the series.
# 

# 
# ## Motivation
# 
# Neel Nanda's post [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems) does a good job explaining the motivation behind solving algorithmic problems such as these. I'd strongly recommend reading the whole post, because it also gives some high-level advice for approaching such problems.
# 
# The main purpose of these challenges isn't to break new ground in mech interp, rather they're designed to help you practice using & develop better understanding for standard MI tools (e.g. interpreting attention, direct logit attribution), and more generally working with libraries like TransformerLens.
# 
# Also, they're hopefully pretty fun, because why shouldn't we have some fun while we're learning?

# ## Logistics
# 
# The deadline is **30th November 2024**. The solution to this problem will be published on this page in the first few days of December, at the same time as the next problem in the sequence. There will also be an associated LessWrong post.
# 
# If you try to interpret this model, you can send your attempt in any of the following formats:
# 
# * Colab notebook,
# * GitHub repo (e.g. with ipynb or markdown file explaining results),
# * Google Doc (with screenshots and explanations),
# * or any other sensible format.
# 
# You should send your attemplt to me (Callum McDougall) via a direct message on Slack (invite link [here](https://join.slack.com/t/arena-uk/shared_invite/zt-2noug8mpy-TRYbCnc3pzj7ITNrZIjKww)) or via email: `cal.s.mcdougall@gmail.com`.
# 
# **I'll feature the names of everyone who sends me a solution on this website, and also give a shout out to the best solutions.**
# 
# Please don't discuss specific things you've found about this model until the challenge is over (although you can discuss general strategies and techniques, and you're also welcome to work in a group if you'd like). The deadline for this problem will be the end of this month, i.e.  30th November.

# ## What counts as a solution?
# 
# Going through the solutions for the previous problems in the sequence as well as the exercises in **[1.5.1] Balanced Bracket Classifier** should give you a good idea of what I'm looking for. In particular, I'd expect you to:
# 
# * Describe a mechanism for how the model solves the task, in the form of the QK and OV circuits of various attention heads (and possibly any other mechanisms the model uses, e.g. the direct path, or nonlinear effects from layernorm),
# * Provide evidence for your mechanism, e.g. with tools like attention plots, targeted ablation / patching, or direct logit attribution.
# * (Optional) Include additional detail, e.g. identifying the subspaces that the model uses for certain forms of information transmission, or using your understanding of the model's behaviour to construct adversarial examples.

# ## Task & Dataset
# 
# The problem for this month is interpreting a model which has been trained to predict the next token in an algorithmically generated sequence. Each sequence has tokens uniformly generated at random, except if the first 2 tokens of a particular trigram appear, in which case the next token is uniquely determined as the third token of the trigram. You can list all the trigrams with `dataset.trigrams`. Here's a demonstration:

# In[ ]:


dataset = BigramDataset(size=10, d_vocab=10, seq_len=10, trigram_prob=0.1, device=device, seed=42)
print(dataset.trigrams)
print(dataset.toks)


# You can see how in this case `(2, 0, 5)` is one of the trigrams, and in both the 4th and 10th sequences above the tokens `(2, 5)` appear consecutively and so must be followed by `5`. Note that the trigrams are generated in a way to make them non-contradictory (i.e. we couldn't have `(2, 0, 5)` and `(2, 0, 6)` in our trigram set).
# 
# The relevant files can be found at:
# 
# ```
# chapter1_transformers/
# └── exercises/
#     └── monthly_algorithmic_problems/
#         └── november24_trigrams/
#             ├── model.py               # code to create the model
#             ├── dataset.py             # code to define the dataset
#             ├── training.py            # code to training the model
#             └── training_model.ipynb   # actual training script
# ```

# ## Model
# 
# The model is **not attention only**. It has one attention layer with a single head, and one MLP layer. It does *not* have layernorm at the end of the model. It was trained with weight decay, and an AdamW optimizer with linearly decaying learning rate. You can load the model in as follows:
# 

# In[ ]:


model = create_model(
    d_vocab=75,
    seq_len=50,
    d_model=32,
    d_head=16,
    n_layers=1,
    n_heads=1,
    d_mlp=20,
    normalization_type=None,
    seed=40,
    device=device,
)

state_dict = t.load(section_dir / "trigram_model.pt", weights_only=True, map_location=device)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False)


# A demonstration of the model working:

# In[ ]:


BIGRAM_PROB = 0.05
BATCH_SIZE = 2500

dataset = BigramDataset(
    size=BATCH_SIZE,
    d_vocab=model.cfg.d_vocab,
    seq_len=model.cfg.n_ctx,
    trigram_prob=BIGRAM_PROB,
    device=device,
    seed=40,
)

logits, cache = model.run_with_cache(dataset.toks)
logprobs = logits[:, :-1].log_softmax(-1)
probs = logprobs.softmax(-1)

targets = dataset.toks[:, 1:]
logprobs_correct = eindex(logprobs, targets, "batch seq [batch seq]")
probs_correct = eindex(probs, targets, "batch seq [batch seq]")

print(f"Average cross entropy loss: {-logprobs_correct.mean().item():.3f}")
print(f"Mean probability on correct label: {probs_correct.mean():.3f}")
print(f"Median probability on correct label: {probs_correct.median():.3f}")
print(f"Min probability on correct label: {probs_correct.min():.3f}")

imshow(
    probs_correct[:50],
    width=600,
    height=600,
    title="Sample model probabilities",
    labels={"x": "Seq posn", "y": "Batch idx"},
)


# You can see from this heatmap that the model is managing to predict the correct token with probability around 100% in a small subset of cases (and you can examine the dataset to see that these are exactly the cases where the two preceding tokens form the start of one of the dataset's trigrams).
# 
# You can also see the Weights & Biases run [here](https://wandb.ai/callum-mcdougall/alg-challenge-trigrams-nov24/runs/c7jjsofv?nw=nwusercallummcdougall). There are 5 important metrics that have been logged:
# 
# - `train_loss`, which is the average cross entropy loss on the training set
# - `train_loss_as_frac`, which is the loss scaled so that 1 is the loss you get when uniformly guessing over all tokens in the vocab, and 0 is the lowest possible loss (where the model has a uniform distribution everywhere except for the trigrams, where it has probability 1 on the correct token)
# - `trigram_*`, which are three metrics specifically for the trigram dataset (i.e. the dataset consisting of only the dataset's special trigrams, i.e. the sequences `(a, b, c)` where `c` always directly follows `ab`). These metrics are only computed on the last token (i.e. the 3rd one) in each sequence. We have:
#     - `trigram_n_correct` = number of trigrams that were correctly predicted
#     - `trigram_frac_correct` = fraction of total trigrams that were correctly predicted
#     - `trigram_avg_correct_prob` = average probability assigned to the correct trigram token
# 
# Note that `trigram_frac_correct` is higher than `trigram_avg_correct_prob`, because some trigrams are predicted with slightly higher than uniform probability but still far from certainty. Also note that neither of these values has hit 100%, indicating that the model has learned most but not all of the trigrams. You can investigate these results for yourself when you inspect the model!
# 
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/wandb-nov24.png" width="1100">

# # Your solution starts here
# 
# 
# Best of luck! 🎈

# ### Summary of how the model works
# 
# The model has one attention layer with a single head, and one MLP layer. It does *not* have layernorm at the end of the model so there are only 3 main components of this model that can be dissected:
# - Embeddings/Unembeddings
#     - Embedding - Converts the input `(seq_len, d_vocab)` to `(seq_len, d_model)` for the inital state of the residual stream containing information about "What features (including sequence position) make up the current token"
#     - Unembedding - Converts the final resiudal stream `(seq_len, d_model)` back into the original input shape `(seq_len, d_vocab)` which represents weighted guesses for what the model thinks will be the next token for each token of the input sequence
#     - These two layers *ALONE* represent the "direct path".
#         - Because the model cannot move information from other tokens, we are simply predicting the next token from the present token. This means that the optimal behavior of these two layers alone is to approximate the bigram log-likelihood.
#         - Example: In a 0 layer transformer, the most likely next token for "Lebron" would be "James" (assuming embedding/unembedding matrices are not weight-tied and model is trained on basketball related text).
# 
# - Attention Layer
#     - Only layer that is capable of moving information between tokens
#     - QK circuit:
#         - Attention patterns show that by default, tokens will have high attention scores with themselves and no other tokens.
#         - EXCEPT in the case, where the the current token is the second token of a "learned trigram" and the token immediately before the current token is the the first token of the same "learned trigram"
#             - Example: say we have the trigram `(7, 8, 9)` and the current sequence is `[4, 7, 1, 9, 8, 2, 8, 8, 34, 67, 54, 12, 87, 7, 8, 9, 12, 19]`.
#             - The QK circuit only has access to information provided by the embedding layer:
#                 - "I am 4, and I am at position x_0"
#                 - "I am 7, and I am at position x_1"
#                 - "I am 1, and I am at postiion x_2"
# 
#                     ...
#                 - "I am 7, and I am at position x_13"
#                 - "I am 8, and I am at position x_14"
#                 - "I am 9, and I am at position x_15"
# 
#                     ...
#                 - "I am 19, and I am at position x_17"
#             - When combining the information from the residual stream with W_Q, all of the `"8s" at postiion x_n` in the sequence will produce a query vector which is asks all of its previous tokens `I'm looking for a 7 at position x_n-1`.
#             - When combining the information from the residual stream with W_K, all of the `"7s" at position x_m` in the sequence will produce a key vector which responds `I'm a 7 at position x_m`.
#             - There is only 1 query/key pair that are a good match for each other and therefore will have a high attention score (the dot product of the key and query vectors is large)
#                 - Query from `token 8 at position x_14`: (`I'm looking for a 7 at position x_13`)
#                 - Key from `token 7 at position x_13`: (`I'm a 7 at postiion x_13`)
#     - OV circuit
#         - copies information about the first token of the bigram to the residual stream of the second token
#         - preserves important information about both bigram tokens (values of both and relative postiions) to be stored in the latter token's residual stream
#         - SVD of the OV circuit shows that the circuit is operating at full rank, meaning it's not doing simple transformations
#     - QV and OV circuits are combined to create a weighted average of the values
#     - Outputs a `(seq_len, d_model)` matrix to be *ADDED* to the residual stream
#     - Using the example above, the resulting residual stream at `position x_14` should have gone from `I am 8 at position x_14` to `I am 8 at position x_14, and I directly follow 7`.
# 
# 
# - MLP Layer
#     - No information is moved/exchanged between tokens
#     - Represent Key-Value memories
#     - Takes the resulting residual stream after the attention layer and performs a key-value lookup (kinda sorta mostly)
#     - For "learned trigrams", will take the bigram prefix as a key and return the value of the third token of the "learned trigram".
#     - Using the above example, the resulting residual stream at `position x_14` should have gone from `I am 8 at position x_14, and I directly follow 7` to `The next token should be 9`.

# 
# ### Helper Functions
# 
# Let's start by defining some helper functions. These will be used to provide evidence for our claims. We can skip these section.

# In[ ]:


from typing import List, Set, Tuple
import torch
import circuitsvis as cv

def detect_trigrams(sequence: torch.Tensor, trigrams: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Detect all trigrams in a sequence.

    Args:
        sequence: torch.Tensor of shape (seq_len,) containing token indices
        trigrams: Set of tuples, each containing three integers representing a trigram

    Returns:
        List of tuples (start_idx, t1, t2, t3) where:
            - start_idx is the position where the trigram starts
            - t1, t2, t3 are the three tokens in the trigram
    """
    found_trigrams = []
    seq_len = len(sequence)

    # Convert sequence to CPU numpy for easier iteration if it's on GPU
    if sequence.device.type == "cuda":
        sequence = sequence.cpu()
    sequence = sequence.numpy()

    # Iterate through sequence looking for trigrams
    for i in range(seq_len - 2):
        # Get the current three tokens
        current_trigram = tuple(sequence[i:i+3])

        # Check if this matches any of our target trigrams
        if current_trigram in trigrams:
            found_trigrams.append((i,) + current_trigram)

    return found_trigrams

def print_trigram_findings(
    sequence: torch.Tensor,
    found_trigrams: List[Tuple[int, int, int, int]],
    context_size: int = 2
) -> None:
    """
    Pretty print the trigrams found in a sequence with surrounding context.

    Args:
        sequence: The original sequence
        found_trigrams: List of (start_idx, t1, t2, t3) tuples from detect_trigrams
        context_size: How many tokens of context to show before/after
    """
    if sequence.device.type == "cuda":
        sequence = sequence.cpu()
    sequence = sequence.numpy()

    print(f"Found {len(found_trigrams)} trigrams:")
    for start_idx, t1, t2, t3 in found_trigrams:
        # Get context boundaries
        context_start = max(0, start_idx - context_size)
        context_end = min(len(sequence), start_idx + 3 + context_size)

        # Get the context tokens
        context = sequence[context_start:context_end]

        # Create the display string with highlighting
        display_tokens = []
        for i, token in enumerate(context):
            pos = i + context_start
            if pos >= start_idx and pos < start_idx + 3:
                display_tokens.append(f"[{token}]")
            else:
                display_tokens.append(str(token))

        print(f"  Position {start_idx}: " + " ".join(display_tokens))
        print(f"    Trigram: {t1} -> {t2} -> {t3}")

def print_trigrams_for_sequence_idx(idx: int) -> None:
  print(f"\nSequence {idx}:")

  length = len(dataset.toks[idx])
  position_tensor = torch.arange(length, device=device)
  print(torch.arange(len(dataset.toks[idx]), device=device))
  print(dataset.toks[idx])

  found = detect_trigrams(dataset.toks[idx], dataset.trigrams)
  print_trigram_findings(dataset.toks[idx], found)

def get_activations(
    model: HookedTransformer, toks: Int[t.Tensor, "batch seq"], names: list[str]
):
    """Uses hooks to return activations from the model, in the form of an ActivationCache."""
    names_list = [names] if isinstance(names, str) else names
    logits, cache = model.run_with_cache(
        toks,
        return_type='logits',
        names_filter=lambda name: name in names_list,
    )
    return logits, cache

def get_out_by_components(model: HookedTransformer, data: BigramDataset):
  '''
  Computes a tensor of shape [3, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
  The first dimension is  [embeddings, head 0.0, mlp 0.0]
  '''
  embedding_hook_names = ["hook_embed", "hook_pos_embed"]
  head_hook_names = [utils.get_act_name("attn_out", layer) for layer in range(model.cfg.n_layers)]
  mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]

  all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
  logits, activations = get_activations(model, data.toks, all_hook_names)

  out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

  for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
      out = t.concat([
          out,
          activations[head_hook_name].unsqueeze(0),
          activations[mlp_hook_name].unsqueeze(0)
      ])

  return logits, out

def calculate_logits_with_attention_pattern(
    model: HookedTransformer,
    cache: ActivationCache,
    pattern: Float[t.Tensor, "batch head_index query_pos key_pos"]
) -> Float[t.Tensor, "batch seq_len vocab_size"]:
    attn_out = calculate_attn_out_from_pattern(model, cache, pattern)
    mlp_out = calculate_mlp_out(model, cache['blocks.0.hook_resid_pre'] + attn_out)
    return model.unembed(cache['blocks.0.hook_resid_pre'] + attn_out + mlp_out)


def calculate_attn_out_from_pattern(
    model: HookedTransformer,
    cache: ActivationCache,
    pattern: Float[t.Tensor, "batch head_index query_pos key_pos"],
) -> Float[t.Tensor, "batch query_pos head_index d_head"]:
    '''
    Copied from TransformerLens:
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/abstract_attention.py#L418-L436
    and
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/abstract_attention.py#L300-L315
    TODO: look into "fwd_hooks"
    '''
    # z = attention pattern * V
    v_ = einops.rearrange(cache['blocks.0.attn.hook_v'], "batch key_pos head_index d_head -> batch head_index key_pos d_head")
    z = einops.rearrange(pattern @ v_, "batch head_index query_pos d_head -> batch query_pos head_index d_head")

    # attn_out = z * W_O
    w = einops.rearrange(model.W_O[0], "head_index d_head d_model -> d_model head_index d_head")
    result = einops.einsum(z, w, "... head_index d_head, d_model head_index d_head -> ... head_index d_model")  # [batch, pos, head_index, d_model]
    return einops.reduce(result, "batch position index model -> batch position model", "sum") + model.b_O[0] # [batch, pos, d_model]

def calculate_mlp_out(
    model: HookedTransformer,
    residual_stream: Float[t.Tensor, "batch seq_len d_model"],
) -> Float[t.Tensor, "batch seq_len d_model"]:
    '''
    Copied from TransformerLens:
    https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/mlps/mlp.py#L47-L49
    TODO: look into "fwd_hooks"
    '''
    pre_act = aaddmm.batch_addmm(model.b_in[0], model.W_in[0], residual_stream)
    post_act = F.relu(pre_act)
    return aaddmm.batch_addmm(model.b_out[0], model.W_out[0], post_act)

def get_top_prediction(
    logits: Float[t.Tensor, "batch seq_len vocab_size"],
    batch_index: int,
    position_index: int
) -> Tuple[float, int]:
    probs = logits[batch_index].softmax(dim=-1)
    confidence_score, predicted_token = t.max(probs[position_index], dim=0)
    return confidence_score.item(), predicted_token.item()



# ### Attention patterns: QK Circuits
# 
# Let's start by examining the attention patterns using the circuitvis library. To start, we are going to use `attention_type="standard"` to analyze the QK circuits.

# In[ ]:


# Convert first 10 sequences to string tokens
str_tokens = [[str(t.item()) for t in seq] for seq in dataset.toks]

# Create labels with full sequences
batch_labels = [f"Sequence #{i}: [{', '.join(str_tokens[i])}]" for i in range(21)]


print_trigrams_for_sequence_idx(0)

cv.attention.from_cache(
    cache = cache,
    tokens = str_tokens,
    batch_idx = list(range(21)),
    attention_type = "standard",
    batch_labels = batch_labels,
)


# If we look at the attention patterns of `Sequence #0`, we see most of the time, each token in the attention head only has high attention scores with itself except for position #6, position #43, and position #49 which have high attenion scores with the token directly before them. If we look closer at position 5-7 `(.., 56, 49, 0, ...)` and position 42-44 `(... ,11, 63, 36, ...)`, we notice that these are both trigrams from the dataset. This tell us that that this attention head is responsible for identifying bigram prefixes for trigrams in the dataset.

# ### Attention Patterns: OV Circuits
# 
# Next, we are going to use `attention_type="info-weighted"`.

# In[ ]:


# Convert first 10 sequences to string tokens
str_tokens = [[str(t.item()) for t in seq] for seq in dataset.toks]

# Create labels with full sequences
batch_labels = [f"Sequence #{i}: [{', '.join(str_tokens[i])}]" for i in range(21)]

print_trigrams_for_sequence_idx(0)


cv.attention.from_cache(
    cache = cache,
    tokens = str_tokens,
    batch_idx = list(range(21)),
    attention_type = "info-weighted",
    batch_labels = batch_labels,
)


# This pretty much looks the same as the previous graph that had `attention_type = "standard"`. This tells us that it's the QK circuit that is making the biggest contribution in overall attention result that gets added to the output stream.
# 
# We can see from this graph that when the attention head recognizes a bigram prefix, it copies information from both the first and latter tokens and stores that information in the latter token's residual stream (to be processed by the mlp layer).
# 
# Why does the last token never have high attention scores with itself and always attend to recent tokens? I couldn't come up with a super convincing answer to me. My best guess is that the model has only been trained on tokens with positions 0, seq_len - 2, so when you ask it to predict the last token at position seq_len - 1, it defaults to attending to recent tokens since the model thinks that gives it the best chance of a correct prediction?

# In[ ]:


import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_svd_single(tensor, title=None):
    U_matrix, S_matrix, V_matrix = t.svd(tensor)

    singular_directions = V_matrix[:, :2]
    # This line of code is changed: we scale with the magnitude of the singular direction
    singular_directions_scaled = utils.to_numpy(singular_directions * S_matrix[:2])
    df = pd.DataFrame(singular_directions_scaled, columns=['Dir 1', 'Dir 2'])
    df['Labels'] = dataset.vocab

    fig = make_subplots(rows=1, cols=2, subplot_titles=["First two singular directions", "Singular values"])
    fig.add_trace(go.Scatter(x=df['Dir 1'], y=df['Dir 2'], mode='markers+text', text=df['Labels']), row=1, col=1)
    fig.update_traces(textposition='top center', marker_size=5)
    fig.add_trace(go.Bar(y=utils.to_numpy(S_matrix)), row=1, col=2)
    fig.update_layout(height=400, width=750, showlegend=False, title_text=title, template="simple_white")
    # Make sure the axes scales are the same (found this code from stackoverflow)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

W_OV = model.W_V[0, 0] @ model.W_O[0, 0] # [d_model, d_model]
W_OV_full = model.W_E @ W_OV # [d_vocab, d_model]

plot_svd_single(W_OV_full.T, title="SVD of W<sub>E</sub>W<sub>OV</sub>")


# Nothing too telling comes from SVD of the W_e * W_ov. The fact that we can't really reduce the dimensions of the OV matrix confirms that the OV circuit is copying unique identifier information about each token to the residual stream.

# ## MLP Layers
# 
# MLP layers are Key-Value memories. For this model, that means that the **key** is the bigram prefix to a trigram and the **value** is the last token in the trigram.
# 
# Let's use a concrete scenario to prove the above claim using sequence #0:
# 
# `tensor([55,  7, 12, 10, 67, 56, 49,  0, 23, 19, 36, 65, 27, 49, 62, 22, 19,  7,
#         38, 50, 70,  8, 18, 41, 66, 67,  9, 72, 43, 37, 36, 47, 69, 51, 61,  5,
#         28,  2, 59, 15, 65, 60, 11, 63, 36,  7, 14,  7, 58,  8])`
# 
# If we look at `positions 20-22`, we have the sequence `70, 8, 18`. This sequence is **NOT** one of the trigrams in the dataset. However, the trigram `22, 8, 38` is one of the trigrams in the dataset and at `position #15`, there is a `22`, the first token of the same trigram. Therefore, if our claim is true and we swap the attention pattern of `position #15` and `position #21`, and run the rest of model as usual, the next token prediction of `position #21` or the next token prediction of `8` should be `38`(the last token of the `22, 8, 38` trigram).
# 
# As mentioned in the QK circuit section, the QK circuit is responsible for identifying bigram prefixes of trigrams in the dataset. So if the attention head identifies that the immediately previous token and the current token are a bigram prefix of one of trigrams learned by the model, the current token will have high attention pattern with immediately previous token. Otherwise, the current token will only have high attention pattern with itself.
# 
# Without manual intervention, the token at `postion #21`, `8`, falls into the latter category of tokens that only attend to themselves. However, after we swap, the attention pattern of `postiion #15 and #21`, the current token will only attend to `position #15`. In other words, by swapping, we have manually edited the contribution of the attention layer at `position #21` to say `I am 8, and I directly follow 22.`

# In[ ]:


logits, cache = model.run_with_cache(
    dataset.toks,
    return_type='logits',
)

batch_index = 0
position_index = 21
swap_index = 15
confidence_score, predicted_token = get_top_prediction(logits=logits, batch_index=batch_index, position_index=position_index)

print(f"Current token at position #{position_index}: {dataset.toks[batch_index][position_index].item()}")
print(f"Next token prediction without swapping: {predicted_token}")
print(f"Probability of next token prediction without swapping: {confidence_score * 100}%")

# Deep clone the attention pattern of batch0, head0
pattern = cache['blocks.0.attn.hook_pattern'][batch_index][0].clone()

# swap the attention pattern of position #15 and postion #21
pattern[position_index][position_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][swap_index]
pattern[position_index][swap_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][position_index]

logits = calculate_logits_with_attention_pattern(model, cache, pattern)
confidence_score, predicted_token = get_top_prediction(logits=logits, batch_index=batch_index, position_index=position_index)


print(f"Manually attending to token at position #{swap_index}: {dataset.toks[batch_index][swap_index].item()}")
print(f"Next token prediction after manual intervention: {predicted_token}")
print(f"Probability of next token prediction after manual intervention: {confidence_score * 100}%")


# From this example, we can see that without swapping the attention scores, the model predicts that the next token that follows `8` at `position #21` is `62` with a confidence score of only 1.48%. This confidence score is barely better than a random guess of 1/75 or 1.33% which means that the model is essentially just guessing a random number.
# 
# However, after swapping attention score, we get the model to predict that the next token that follows `8` at `position #21` is `38` with close to a 100% confidence score which confirms our inital hypothesis.

# Here is another example where I was able to manually change the attention pattern to get the model to predict the last token of two trigrams that share the same second token. `(68, 45, 30)` and `(51, 45, 67)`

# In[ ]:


logits, cache = model.run_with_cache(
    dataset.toks,
    return_type='logits',
)

batch_index = 69
position_index = 39
swap_index = 28
confidence_score, predicted_token = get_top_prediction(logits=logits, batch_index=batch_index, position_index=position_index)

print(f"Current token at position #{position_index}: {dataset.toks[batch_index][position_index].item()}")
print(f"Next token prediction without swapping: {predicted_token}")
print(f"Probability of next token prediction without swapping: {confidence_score * 100}%")

# Deep clone the attention pattern
pattern = cache['blocks.0.attn.hook_pattern'][batch_index][0].clone()

# swap the attention pattern
pattern[position_index][position_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][swap_index]
pattern[position_index][swap_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][position_index]

logits = calculate_logits_with_attention_pattern(model, cache, pattern)
confidence_score, predicted_token = get_top_prediction(logits=logits, batch_index=batch_index, position_index=position_index)

print()
print(f"Manually attending to token at position #{swap_index}: {dataset.toks[batch_index][swap_index].item()}")
print(f"Next token prediction after manual intervention: {predicted_token}")
print(f"Probability of next token prediction after manual intervention: {confidence_score * 100}%")

swap_index = 16

# Deep clone the attention pattern
pattern = cache['blocks.0.attn.hook_pattern'][batch_index][0].clone()

# swap the attention pattern
pattern[position_index][position_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][swap_index]
pattern[position_index][swap_index] = cache['blocks.0.attn.hook_pattern'][batch_index][0][position_index][position_index]

logits = calculate_logits_with_attention_pattern(model, cache, pattern)
confidence_score, predicted_token = get_top_prediction(logits=logits, batch_index=batch_index, position_index=position_index)

print()
print(f"Manually attending to token at position #{swap_index}: {dataset.toks[batch_index][swap_index].item()}")
print(f"Next token prediction after manual intervention: {predicted_token}")
print(f"Probability of next token prediction after manual intervention: {confidence_score * 100}%")


# Note: I was able to get this trick of swapping attention to work most of the time, but not all the time and my best guess is that it has something to do with increased interference from the positional embeddings if the source and destination tokens are too far away.

# ## Abalation Study

# In[ ]:


def print_stuff(out, model, logits):

  context_size: int = 2
  count_correctly_pred_trigrams = 0;
  count_correctly_pred_trigrams_direct = 0;
  count_correctly_pred_trigrams_attn = 0;
  count_correctly_pred_trigrams_mlp = 0;
  count_logits_correctly_pred_trigrams = 0;
  total_trigrams = 0;

  set_of_trigrams_found = set()

  set_of_trigrams_found_in_direct_path = set()

  mlp_out_without_attn = calculate_mlp_out(model, out[0, :, :, :])

  for batch_idx in range(out.shape[1]):
  # for batch_idx in range(1):

    # sequence = dataset.toks[batch_idx]

    found_trigrams = detect_trigrams(dataset.toks[batch_idx], dataset.trigrams)
    total_trigrams += len(found_trigrams)

    # print(f"Found {len(found_trigrams)} trigrams in Sequence {batch_idx}:")
    for start_idx, t1, t2, t3 in found_trigrams:
        seq_pos = start_idx + 1

        # # Get context boundaries
        # context_start = max(0, start_idx - context_size)
        # context_end = min(len(sequence), start_idx + 3 + context_size)

        # # Get the context tokens
        # context = sequence[context_start:context_end]

        # # Create the display string with highlighting
        # display_tokens = []
        # for i, token in enumerate(context):
        #     pos = i + context_start
        #     if pos >= start_idx and pos < start_idx + 3:
        #         display_tokens.append(f"[{token}]")
        #     else:
        #         display_tokens.append(str(token))

        # print(f"  Position {start_idx}: " + " ".join(display_tokens))
        # print(f"    Trigram: {t1} -> {t2} -> {t3}")

        batch_outputs_by_components = out[:, batch_idx, :, :] # [3, seq_pos, d_model]

        sum = torch.sum(batch_outputs_by_components[[0,1,2]], dim=0)
        logits_from_resid = model.unembed(sum) # [seq_pos, d_vocab]
        probs_from_resid = logits_from_resid.softmax(dim=-1)
        confidence_score, predicted_token = torch.max(probs_from_resid[seq_pos], dim=0)

        if predicted_token.item() == t3.item():
          count_correctly_pred_trigrams += 1
          set_of_trigrams_found.add((t1, t2, t3))

        max_value1, max_index1 = torch.max(logits[batch_idx, seq_pos], dim=0)
        if (max_index1.item() == t3.item()):
          count_logits_correctly_pred_trigrams += 1

        # direct path
        # why does the direct path have accuracy that is so much higher than random sampling?
        logits_from_direct_path = model.unembed(batch_outputs_by_components[0]) # [seq_pos, d_vocab]
        prob_from_direct_path = logits_from_direct_path.softmax(dim=-1)
        confidence_score, predicted_token = torch.max(prob_from_direct_path[seq_pos], dim=0)
        if predicted_token.item() == t3.item():
          count_correctly_pred_trigrams_direct += 1
          set_of_trigrams_found_in_direct_path.add((t2, t3))

        # without attention layer
        logits_from_resid_without_attn = model.unembed(mlp_out_without_attn[batch_idx][seq_pos])
        prob_from_direct_without_attn = logits_from_resid_without_attn.softmax(dim=-1)
        confidence_score, predicted_token = torch.max(prob_from_direct_without_attn[seq_pos], dim=0)
        if predicted_token.item() == t3.item():
          count_correctly_pred_trigrams_attn += 1


        # without mlp layer
        logits_from_resid_without_mlp = model.unembed(torch.sum(batch_outputs_by_components[[0,1]], dim=0)) # [seq_pos, d_vocab]
        logits_from_resid_without_mlp = logits_from_resid_without_mlp.softmax(dim=-1)
        confidence_score, predicted_token = torch.max(logits_from_resid_without_mlp[seq_pos], dim=0)
        if predicted_token.item() == t3.item():
          count_correctly_pred_trigrams_mlp += 1

  print("total trigrams detected ('manually'): " + str(total_trigrams))


  print("Trigrams (correctly predicted, total, percentage)")
  print("correctly predicted trigrams: " + str(count_correctly_pred_trigrams))
  print(count_correctly_pred_trigrams / total_trigrams)

  print("Trigrams with zero-ablation (correctly predicted, total, percentage)")
  print("correctly predicted trigrams direct path: " + str(count_correctly_pred_trigrams_direct))
  print("correctly predicted trigrams only attn: " + str(count_correctly_pred_trigrams_mlp))
  print("correctly predicted trigrams only mlp: " + str(count_correctly_pred_trigrams_attn))

  print("accuracy direct path: " + str(count_correctly_pred_trigrams_direct / total_trigrams) + "%")
  print("accuracy only attn: " + str(count_correctly_pred_trigrams_mlp / total_trigrams) + "%")
  print("accuracy only mlp: " + str(count_correctly_pred_trigrams_attn / total_trigrams) + "%")



  print("Trigrams (learned, total, percentage)")
  print(len(set_of_trigrams_found))
  print(set_of_trigrams_found)
  print(len(dataset.trigrams))
  print(len(set_of_trigrams_found) / len(dataset.trigrams))

  print("Absolute truth")
  print("correctly predicted trigrams: " + str(count_logits_correctly_pred_trigrams))
  print("accuracy: " + str(count_logits_correctly_pred_trigrams / total_trigrams * 100) + "%")


  print("Direct path")
  print(set_of_trigrams_found_in_direct_path)
  print(len(set_of_trigrams_found_in_direct_path))

  return set_of_trigrams_found_in_direct_path



logits, out = get_out_by_components(model, dataset)

set_of_trigrams_found_in_direct_path = print_stuff(out, model, logits)

