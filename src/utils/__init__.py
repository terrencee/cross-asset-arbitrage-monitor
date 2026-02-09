"""
Utility functions
"""

# Import with error handling for logger
try:
    from .logger import log, setup_logger
except:
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    setup_logger = None

from .validators import DataValidator

__all__ = [
    'log',
    'setup_logger',
    'DataValidator'
]
