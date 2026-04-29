from torch.nn import Module, LayerNorm, GELU
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules import mamba2

class ResidualMamba(Module):
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super(ResidualMamba, self).__init__()

        self.layer_norm = LayerNorm(d_model)

        self.mamba = Mamba(d_model, d_state, d_conv, expand)


    def forward(self, x):

        residual = x
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = GELU()(x)

        return x + residual

