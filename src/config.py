"""File containing basic project configurations"""

# Dependencies
import os
from pathlib import Path

# Get application root directory (where main.py is located)
ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_NAME = 'resource_allocation_algo'

CONFIG = {
    # Database configuration
    'use_db': False,
    'db_url': f"sqlite:///{os.path.join(ROOT_DIR, 'data', 'production.db')}",

    'empty_dataframes': ['employee_shift_assignments_table', 'product_production_line_assignments_table'],
    
    # Base directories
    'data_dir': os.path.join(ROOT_DIR, 'data'),
    'output_dir': os.path.join(ROOT_DIR, 'data', 'output'),
    'log_dir': os.path.join(ROOT_DIR, 'logs'),
    
    # File paths mapping
    'filepath_map': {
        'contract_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'contracttype_table.csv'),
        'demands': os.path.join(ROOT_DIR, 'data', 'csvs', 'demand_table.csv'),
        'days_numbers': os.path.join(ROOT_DIR, 'data', 'csvs', 'daysnumber_table.csv'),
        'employees': os.path.join(ROOT_DIR, 'data', 'csvs', 'employee_table.csv'),
        'employee_hours': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeehours_table.csv'),
        'employee_production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeeproductionline_table.csv'),
        'employee_shift_assignments': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeeshiftassignment_table.csv'),
        'groups': os.path.join(ROOT_DIR, 'data', 'csvs', 'group_table.csv'),
        'line_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'linetype_table.csv'),
        'operating_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'operatingtype_table.csv'),
        'products': os.path.join(ROOT_DIR, 'data', 'csvs', 'product_table.csv'),
        'production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionline_table.csv'),
        'production_lines_stats': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionlinestats_table.csv'),
        'production_lines_operating_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionlineoperatingtype_table.csv'),
        'product_production_line_assignments': os.path.join(ROOT_DIR, 'data', 'csvs', 'productproductionlineassignment_table.csv'),
        'product_production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'productproductionline_table.csv'),
        'sections': os.path.join(ROOT_DIR, 'data', 'csvs', 'section_table.csv'),
        'shifts': os.path.join(ROOT_DIR, 'data', 'csvs', 'shift_table.csv'),
        'shift_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'shifttype_table.csv')
    },

    # File paths
    'dummy_data_filepaths': {
        'contract_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'contracttype_table.csv'),
        'demands': os.path.join(ROOT_DIR, 'data', 'csvs', 'demand_table.csv'),
        'days_numbers': os.path.join(ROOT_DIR, 'data', 'csvs', 'daysnumber_table.csv'),
        'employees': os.path.join(ROOT_DIR, 'data', 'csvs', 'employee_table.csv'),
        'employee_hours': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeehours_table.csv'),
        'employee_production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeeproductionline_table.csv'),
        'employee_shift_assignments': os.path.join(ROOT_DIR, 'data', 'csvs', 'employeeshiftassignment_table.csv'),
        'groups': os.path.join(ROOT_DIR, 'data', 'csvs', 'group_table.csv'),
        'line_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'linetype_table.csv'),
        'operating_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'operatingtype_table.csv'),
        'products': os.path.join(ROOT_DIR, 'data', 'csvs', 'product_table.csv'),
        'production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionline_table.csv'),
        'production_lines_stats': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionlinestats_table.csv'),
        'production_lines_operating_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'productionlineoperatingtype_table.csv'),
        'product_production_line_assignments': os.path.join(ROOT_DIR, 'data', 'csvs', 'productproductionlineassignment_table.csv'),
        'product_production_lines': os.path.join(ROOT_DIR, 'data', 'csvs', 'productproductionline_table.csv'),
        'sections': os.path.join(ROOT_DIR, 'data', 'csvs', 'section_table.csv'),
        'shifts': os.path.join(ROOT_DIR, 'data', 'csvs', 'shift_table.csv'),
        'shift_types': os.path.join(ROOT_DIR, 'data', 'csvs', 'shifttype_table.csv')
    },
    
    # Available algorithms (used by AlgorithmFactory)
    'available_algorithms': ['fillbags', 'lp'],
    
    # Process configuration
    'stages': {
        'data_loading': {
            'sequence': 1,
            'requires_previous': False,
            'validation_required': True,
            'decisions': {
                'selections': {
                    'apply_selection': True,
                    'months': [1],
                    'years': [2024]
                }
            }
        },
        'data_transformation': {
            'sequence': 2,
            'requires_previous': True,
            'validation_required': True,
            'decisions': {
                'filtering':{
                    'apply_filtering': False,
                    'excluded_employees': '', # to create defaults, try using a list, and make sure the decision handling is not convertng it to a string
                    'excluded_lines': '', # to create defaults, try using a list, and make sure the decision handling is not convertng it to a string
                },
                'time_periods': 1
            }
        },
        'product_allocation': {
            'sequence': 3,
            'requires_previous': True,
            'validation_required': True,
            'decisions': {
                'product_assignments': {
                    'product_id': [],
                    'production_line_id': [],
                    'quantity': []
                }
            }
        },
        'resource_allocation': {
            'sequence': 4,  
            'requires_previous': True,
            'validation_required': True,
            'decisions': {
                'algorithms': ['fillbags', 'lp'] # TODO: add 'decisions' hierarchy as the others stages
            }
        },
        'result_analysis': {
            'sequence': 5,
            'requires_previous': True,
            'validation_required': True,
            'decisions': {
                'changes': {
                    'add_changes': False,
                    'special_allocations': {}
                },
                'generate_report': False
            }
        }
    },
    
    # Algorithm parameters
    'algorithm_defaults': {
        'fillbags': {
            'sort_strategy': 'by_colors',
            'prioritize_high_capacity': True
        },
        'lp': {
            'temporal_space': 1,
            'objective_weights': {
                'understaffing': 1.0,
                'overstaffing': 1.0
            }
        }
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