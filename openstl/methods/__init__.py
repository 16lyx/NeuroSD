# openstl/methods/__init__.py

from .MTFormer import MTformer
from .neurosd import NeuroSD

method_maps = {
    'mtformer': MTformer,
    'neurosd': NeuroSD,
}

__all__ = ['method_maps', 'MTformer', 'NeuroSD']

# BEGIN NS30_OFFICIAL_BASELINE_METHODS
from .convlstm import ConvLSTM
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .e3dlstm import E3DLSTM
from .mim import MIM
from .mau import MAU
from .phydnet import PhyDNet
from .prednet import PredNet
from .simvp import SimVP
from .tau import TAU

if 'method_maps' not in globals():
    method_maps = {}

method_maps.update({
    'convlstm': ConvLSTM,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'e3dlstm': E3DLSTM,
    'mim': MIM,
    'mau': MAU,
    'phydnet': PhyDNet,
    'prednet': PredNet,
    'simvp': SimVP,
    'tau': TAU,
})

if '__all__' in globals():
    __all__ = list(dict.fromkeys(__all__ + ['method_maps', 'ConvLSTM', 'PredRNN', 'PredRNNpp', 'PredRNNv2', 'E3DLSTM', 'MIM', 'MAU', 'PhyDNet', 'PredNet', 'SimVP', 'TAU']))
# END NS30_OFFICIAL_BASELINE_METHODS

# BEGIN NS30_EXTRA_BASELINES
from .ns30_extra import NS30ExtraBaseline
method_maps.update({name: NS30ExtraBaseline for name in ('resnet', 'unet', 'vit', 'cno', 'fno', 'mmvp')})
# END NS30_EXTRA_BASELINES

# NS30_EARTHFORMER
method_maps['earthformer'] = NS30ExtraBaseline

from .predformer import PredFormer

method_maps.update({
    'predformer': PredFormer,
    'PredFormer': PredFormer,
})
