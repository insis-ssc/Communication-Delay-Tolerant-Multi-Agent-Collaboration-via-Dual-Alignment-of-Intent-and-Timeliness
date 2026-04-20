from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .maic_learner import MAICLearner
from .maic_qplex_learner import MAICQPLEXLearner
from .maddpg_learner import MADDPGLearner
from .Code_learner import Code_learner
from .Code_seq_learner import Code_seq_learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY['maic_learner'] = MAICLearner
REGISTRY['maic_qplex_learner'] = MAICQPLEXLearner
REGISTRY['Code'] = Code_learner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["Code_seq"] = Code_seq_learner