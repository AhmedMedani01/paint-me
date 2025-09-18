"""
Configuration settings for the Hexa Paint Animation backend
"""

import os

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hexa_paint_secret_key_2024'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'outputs'
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'usage_stats.db'
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    
    # Animation parameters
    ANIMATION_PARAMS = {
        'dilate_iters': 1,
        'kernel_size': 3,
        'min_size': 1500,
        'closing_size': 5,
        'skip': 100,
        'reverse': False,
        'ngroups': 4,
        'hex_radius': 30,
        'jitter': 0.3,
        'rng_seed': 42
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = ':memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
