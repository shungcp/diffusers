import jax.numpy as jnp
import torch.nn.functional as F
import time
import torch
import torchax
import jax

torchax.enable_globally()


size = 50000


a = torch.randn((size, size), dtype=torch.bfloat16, device='jax')
b = torch.randn((size, size), dtype=torch.bfloat16, device='jax')

for i in range(3):
  start = time.perf_counter() 
  jax.lax.dot(a.jax(), b.jax(), preferred_element_type=jnp.bfloat16).block_until_ready()
  end = time.perf_counter()
  print(i, end - start)

c = torch.randn((size, size), dtype=torch.int8, device='jax')
d = torch.randn((size, size), dtype=torch.bfloat16, device='jax')

print(' === int8 ')

for i in range(3):
  start = time.perf_counter() 
  jax.lax.dot(c.jax(), d.jax(), preferred_element_type=jnp.bfloat16).block_until_ready()
  end = time.perf_counter()
  print(i, end - start)