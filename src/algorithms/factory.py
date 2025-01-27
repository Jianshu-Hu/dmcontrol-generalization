from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.eda import EDA
from algorithms.drc import DRC
from algorithms.wro import WRO
from algorithms.dro import DRO
from algorithms.esac import ESAC
from algorithms.edrq import EDrQ
from algorithms.esvea import ESVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
	'eda': EDA,
	'drc': DRC,
	'wro': WRO,
	'dro': DRO,
	'esac': ESAC,
	'edrq': EDrQ,
	'esvea': ESVEA
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
