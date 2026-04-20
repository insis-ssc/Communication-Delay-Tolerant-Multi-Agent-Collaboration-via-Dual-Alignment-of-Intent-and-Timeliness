REGISTRY = {}

from .basic_controller import BasicMAC
from .maic_controller import MAICMAC
from .Code_controller import Code_Controller
from .maddpg_controller import MADDPGMAC
from .Code_controller_seq import Code_Controller_Seq

REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY['maic_mac'] = MAICMAC
REGISTRY['Code'] = Code_Controller
REGISTRY['Code_seq'] = Code_Controller_Seq