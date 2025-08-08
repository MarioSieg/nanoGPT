import multiprocessing
import time

import torch
import tiktoken
from model import GPT

max_new_tokens = 64
temperature = 0.9
seed = 1337
device = 'cpu'

torch.manual_seed(seed)
torch.set_default_device(device)
#torch.set_num_threads(multiprocessing.cpu_count())

print(torch.get_num_threads())
print(torch.get_num_interop_threads())

model = GPT.from_pretrained('gpt2', dict(dropout=0.0))

model.eval()
model = model.float()

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start = 'What is the answer to life?'
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    start = time.perf_counter()
    y = model.generate(x, max_new_tokens, temperature=temperature)
    elapsed = time.perf_counter() - start
    print(f"Generated in: {elapsed:.9f} seconds")
    print(decode(y[0].tolist()))
