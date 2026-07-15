"""Demo: attributo normale vs register_buffer vs nn.Parameter.

Non serve una GPU: model.to(torch.float64) usa lo stesso meccanismo di
model.to("cuda") — converte solo cio' che il modulo conosce.
"""
import torch
from torch import nn


class Demo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plain = torch.ones(2, 2)                      # attributo normale
        self.register_buffer("buffer", torch.ones(2, 2))   # buffer
        self.param = nn.Parameter(torch.ones(2, 2))        # parametro


model = Demo()
model.to(torch.float64)  # stesso meccanismo di .to("cuda")

print("dopo model.to(float64):")
print("  plain  ->", model.plain.dtype, " (rimasto indietro!)")
print("  buffer ->", model.buffer.dtype)
print("  param  ->", model.param.dtype)

print("nello state_dict (= nel checkpoint):", list(model.state_dict()))

# E se l'adiacenza fosse un Parameter? L'optimizer la "allena".
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
loss = (model.param * torch.randn(2, 2)).sum()
loss.backward()
optimizer.step()
print("param dopo 1 optimizer step (non e' piu' la matrice 0/1!):")
print(model.param.data)
print("buffer dopo lo stesso step (intatto):")
print(model.buffer)
