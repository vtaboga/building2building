import logging
import os
from datetime import datetime

def setup_logger(name: str, filename: str = None, add_handlers: bool = True) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        filename (str): Name of the log file
        add_handlers (bool): Whether to add handlers to the logger
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Only add handlers if requested and if the logger doesn't already have handlers
    if add_handlers and not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if filename is provided)
        if filename:
            file_handler = logging.FileHandler(os.path.join('logs', filename))
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger 
