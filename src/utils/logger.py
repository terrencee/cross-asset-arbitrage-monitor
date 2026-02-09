"""
Logging utility for the arbitrage monitor
Provides consistent logging across all modules
"""

from loguru import logger
import sys
from pathlib import Path

def setup_logger(config_path: str = "config.yaml", use_file: bool = True):
    """
    Setup logger with configuration from config file
    
    Args:
        config_path: Path to configuration YAML file
        use_file: Whether to log to file (default True)
    """
    # Default configuration
    log_config = {
        'level': 'INFO',
        'file': 'logs/arbitrage_monitor.log',
        'rotation': '1 day',
        'retention': '30 days'
    }
    
    # Try to load config from file
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_config.update(config.get('logging', {}))
    except (FileNotFoundError, ImportError):
        pass  # Use defaults
    
    # Remove default logger
    logger.remove()
    
    # Add console output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_config.get('level', 'INFO')
    )
    
    # Add file output if requested
    if use_file:
        try:
            log_file = Path(log_config.get('file', 'logs/arbitrage_monitor.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                rotation=log_config.get('rotation', '1 day'),
                retention=log_config.get('retention', '30 days'),
                level=log_config.get('level', 'INFO'),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
            )
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    logger.info("Logger initialized successfully")
    return logger

# Create global logger instance
log = setup_logger()
