"""Configuration for the my_new_project project."""

import os
from pathlib import Path

# Project name - used for logging and process tracking
PROJECT_NAME = "my_new_project"

# Get application root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG = {
    # Database configuration
    'use_db': False,
    'db_url': f"sqlite:///{os.path.join(ROOT_DIR, 'data', 'production.db')}",
    
    # Base directories
    'data_dir': os.path.join(ROOT_DIR, 'data'),
    'output_dir': os.path.join(ROOT_DIR, 'data', 'output'),
    'log_dir': os.path.join(ROOT_DIR, 'logs'),
    
    # File paths for CSV data sources
    'dummy_data_filepaths': {
        # Example data files mapping - replace with your actual data files
        'customers': os.path.join(ROOT_DIR, 'data', 'csvs', 'customers.csv'),
        'transactions': os.path.join(ROOT_DIR, 'data', 'csvs', 'transactions.csv'),
        'products': os.path.join(ROOT_DIR, 'data', 'csvs', 'products.csv'),
    },
    
    # Available algorithms for the project
    'available_algorithms': [
        'example_algorithm',
        # Add your custom algorithms here
    ],
    
    # Process configuration - stages and decision points
    'stages': {
        # Stage 1: Data Loading
        'data_loading': {
            'sequence': 1,               # Stage order
            'requires_previous': False,  # First stage doesn't require previous stages
            'validation_required': True, # Validate data after loading
            'decisions': {
                'selections': {          # Decision point for data selection
                    'selected_entities': ['customers', 'transactions'],  # Default entities to load
                    'load_all': False,   # Whether to load all available entities
                }
            }
        },
        
        # Stage 2: Data Transformation
        'data_transformation': {
            'sequence': 2,
            'requires_previous': True,   # Requires previous stage completion
            'validation_required': True,
            'decisions': {
                'transformations': {     # Decision point for transformation options
                    'apply_filtering': False,
                    'filter_column': '',
                    'filter_value': '',
                    'normalize_numeric': True,  # Whether to normalize numerical data
                    'fill_missing': True,       # Whether to fill missing values
                    'fill_method': 'mean'       # Method for filling missing values
                }
            }
        },
        
        # Stage 3: Processing (Algorithm Execution)
        'processing': {
            'sequence': 3,
            'requires_previous': True,
            'validation_required': True,
            'algorithms': [              # Available algorithms for this stage
                'example_algorithm'
            ],
            'decisions': {
                'algorithm_selection': {  # Decision point for algorithm selection
                    'algorithm': 'example_algorithm',
                    'parameters': {       # Default algorithm parameters
                        'threshold': 50.0,
                        'include_outliers': False,
                        'outlier_threshold': 2.0
                    }
                }
            }
        },
        
        # Stage 4: Result Analysis
        'result_analysis': {
            'sequence': 4,
            'requires_previous': True,
            'validation_required': True,
            'decisions': {
                'report_generation': {    # Decision point for result reporting
                    'generate_report': True,
                    'report_format': 'csv',
                    'include_visualizations': False,
                    'save_detailed_results': True
                }
            }
        }
    },
    
    # Algorithm parameters (defaults for each algorithm)
    'algorithm_defaults': {
        'example_algorithm': {
            'threshold': 50.0,
            'include_outliers': False,
            'outlier_threshold': 2.0
        },
        # Add defaults for your custom algorithms here
    },
    
    # Output configuration
    'output': {
        'base_dir': 'data/output',
        'visualizations_dir': 'data/output/visualizations',
        'diagnostics_dir': 'data/diagnostics'
    },
    
    # Logging configuration
    'log_level': 'INFO',
    'log_format': '%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s',
    'log_dir': 'logs'
}

# Add any project-specific configuration below