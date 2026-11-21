from .MTformer import MTformer_Model


__all__ = [
    'MTformer_Model'
]

# BEGIN NS30_OFFICIAL_BASELINE_MODELS
from .convlstm_model import ConvLSTM_Model
from .e3dlstm_model import E3DLSTM_Model
from .mau_model import MAU_Model
from .mim_model import MIM_Model
from .phydnet_model import PhyDNet_Model
from .prednet_model import PredNet_Model
from .predrnn_model import PredRNN_Model
from .predrnnpp_model import PredRNNpp_Model
from .predrnnv2_model import PredRNNv2_Model
from .simvp_model import SimVP_Model

if '__all__' in globals():
    __all__ = list(dict.fromkeys(__all__ + ['ConvLSTM_Model', 'E3DLSTM_Model', 'MAU_Model', 'MIM_Model', 'PhyDNet_Model', 'PredNet_Model', 'PredRNN_Model', 'PredRNNpp_Model', 'PredRNNv2_Model', 'SimVP_Model']))
# END NS30_OFFICIAL_BASELINE_MODELS
