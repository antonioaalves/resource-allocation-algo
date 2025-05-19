"""File containing the class AlgorithmFactory"""

# Dependencies
import logging
from typing import Optional, Dict, Any

# Local stuff
from base_data_project.algorithms.base import BaseAlgorithm
from src.algorithms.lp_algo import LpAlgo
from src.algorithms.fill_bags import FillBagsAlgorithm

logger = logging.getLogger('BagAllocationAlgo')

class AlgorithmFactory:
    """
    Factory class for creating algorithm instances
    """

    @staticmethod
    def create_algorithm(decision: str, parameters: Optional[Dict[str, Any]] = None) -> BaseAlgorithm:
        """Choose an algorithm based on user decisions"""

        if parameters is None:
            # Use default configuration if not provided 
            from src.config import CONFIG
            parameters = {
                'available_algos': CONFIG.get('available_algorithms')
            }

        if decision.lower() not in parameters.get('available_algos'):
            # If decision not available, raise an exception
            msg = f"Decision made for algorithm selection not available in config file config.py"
            logger.error(msg)
            raise ValueError(msg)

        if decision.lower() == 'LpAlgo':
            logger.info()
            return LpAlgo(parameters=parameters)
        elif decision.lower() == 'FillBagsAlgorithm':
            logger.info()
            return FillBagsAlgorithm(parameters=parameters)
        else:
            error_msg = f"Unsupported algorithm type: {decision}"
            logger.error(error_msg)
            raise ValueError(error_msg)