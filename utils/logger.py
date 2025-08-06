import logging
import os
from datetime import datetime
from typing import Optional

class CounterLogger:
    def __init__(self, name: str, log_dir: str = "logs", level: str = "INFO"):
        self.name = name
        self.log_dir = log_dir
        self.logger = self._setup_logger(level)
    
    def _setup_logger(self, level: str) -> logging.Logger:
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if logger.handlers:
            return logger
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message) 