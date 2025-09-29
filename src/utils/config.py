"""Configuration management utilities."""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .logger import get_logger
logger = get_logger(__name__)

class ConfigManager:
    """Singleton configuration manager."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance  
    
    def load_config(self, config_path: str = "config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""

        if self._config is not None:
            return self._config
        
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        self._create_directories()
        
        return self._config
    
    def _create_directories(self) -> None:
        """Create necessary directories based on configuration."""

        if self._config is None:
            return
        
        # Create data directories
        for key, path in self._config.get('data', {}).items():
            if 'dir' in key:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value."""

        if self._config is None:
            self.load_config()
        
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Updated config: {key} = {value}")
    
    def save_config(self, config_path: str = "config.yaml") -> None:
        """Save current configuration to file."""

        if self._config is None:
            logger.warning("No configuration to save")
            return
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        
        if self._config is None:
            self.load_config()
        return self._config
    
    def reset(self) -> None:
        """Reset configuration (mainly for testing)."""
        self._config = None


# Global config instance
config = ConfigManager()