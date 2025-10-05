import warnings
try:
    import rdkit
except ImportError:
    HAS_RDKIT = False
    warnings.warn("RDKit is not installed. dl_util.py will be unavailable.")
else:
    HAS_RDKIT = True

try:
    import pynvml
except ImportError:
    HAS_PYNVML = False
    warnings.warn("pynvml is not installed. GPU memory usage will be unavailable.")
else:
    HAS_PYNVML = True

try:
    import sklearn
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn is not installed. KFold will be unavailable.")
else:
    HAS_SKLEARN = True


from .registry import registry
from ..logging import Logger
from .util import now, Namespace, set_seed, set_proxy, better_dict_4_print
from .io import load, dump, get_file_size, makedirs, cleanup_dir
from ..util.dl_util import get_model_params_num, freeze_network, mean_pooling


# if HAS_RDKIT:
#     from .bio_util import generate_mol_img
if HAS_PYNVML:
    from ..util.dl_util import get_gpu_usage
if HAS_SKLEARN:
    from ..util.dl_util import kfold


