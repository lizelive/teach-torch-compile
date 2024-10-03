

from typing import Final
import torch
import torch.nn.functional as F
from torch import Tensor

def diy_conv2d(inp: Tensor, w: Tensor):
    kernel_size = ( w.shape[-2], w.shape[-1]) 
    inp_unf = torch.nn.functional.unfold(inp, kernel_size)
    out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
    return out


def test_diy_conv2d():
    inp = torch.ones(1, 3, 10, 12)
    w = torch.ones(2, 3, 4, 5)

    out = diy_conv2d(inp, w)
    gt = torch.nn.functional.conv2d(inp, w)

    assert torch.allclose(gt, out)

# test_diy_conv2d()

class DiyConv2d(torch.nn.Module):
    w: Final[Tensor]
    def __init__(self, w: Tensor):
        self.w = w
        super(DiyConv2d, self).__init__()

    def forward(self, inp: Tensor) -> Tensor:
        return diy_conv2d(inp, self.w)
    
model_params = (torch.ones(2, 3, 4, 5), )
model_args = (torch.ones(1, 3, 10, 12), )


mod = DiyConv2d(*model_params)
exported = torch.export.export(mod, args =  model_args)

print(exported)
# print(torch.jit.script(exported))
torch.onnx.export(mod, args = model_args, f = 'conv.onnx', opset_version=20)
