# my_new_project

A data processing project built with the Base Data Project framework.

## Overview

This project provides a standardized approach to data processing, with built-in support for:
- Consistent data management from different sources
- Multi-stage process execution with decision tracking
- Algorithm implementation using standard patterns
- Progress monitoring and logging

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure the project:
   - Edit `src/config.py` to set project-specific configurations
   - Add custom algorithms in `src/algorithms/`
   - Configure process stages and decisions in `src/config.py`

## Project Structure

```
my_new_project/
├── main.py                # Main entry point for interactive mode
├── batch_process.py       # Batch processing script
├── routes.py              # API server for HTTP access
├── data/                  # Data directory
│   ├── csvs/              # Input CSV files
│   └── output/            # Output files
├── logs/                  # Log files
└── src/                   # Source code
    ├── __init__.py
    ├── config.py          # Project configuration
    ├── algorithms/        # Custom algorithms
    │   ├── __init__.py
    │   └── example_algorithm.py
    └── services/          # Business logic services
        ├── __init__.py
        └── example_service.py
```

## Running the Project

### Interactive Mode

Interactive mode provides a step-by-step interface with user prompts:

```bash
python main.py run-process
```

This will guide you through the process stages with interactive decision-making.

### Batch Mode

Batch mode runs the entire process non-interactively:

```bash
python batch_process.py --algorithm example_algorithm
```

Options:
- `--use-db`: Use database instead of CSV files
- `--no-tracking`: Disable process tracking
- `--algorithm`: Specify which algorithm to use

### API Server (Optional)

To run the API server for HTTP access:

```bash
python routes.py
```

This starts a Flask server with endpoints for accessing the functionality.

## Development

### Adding a New Algorithm

1. Create a new file in `src/algorithms/` (e.g., `my_algorithm.py`)
2. Inherit from `base_data_project.algorithms.base.BaseAlgorithm`
3. Implement required methods:
   - `adapt_data`: Transform input data for processing
   - `execute_algorithm`: Implement core algorithm logic
   - `format_results`: Structure the outputs consistently
4. Register in `src/algorithms/__init__.py`
5. Add to available algorithms in `src/config.py`

Example:

```python
from base_data_project.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def adapt_data(self, data=None):
        # Transform input data for processing
        return transformed_data
        
    def execute_algorithm(self, adapted_data=None):
        # Implement core algorithm logic
        return results
        
    def format_results(self, algorithm_results=None):
        # Structure outputs consistently
        return formatted_results
```

### Creating a Custom Service

Services coordinate data management, process tracking, and algorithm execution:

1. Create a new file in `src/services/` (e.g., `my_service.py`)
2. Implement service methods to handle:
   - Process initialization
   - Stage execution
   - Decision management
   - Result handling

Example:

```python
from src.config import PROJECT_NAME

class MyService:
    def __init__(self, data_manager, process_manager=None):
        self.data_manager = data_manager
        self.process_manager = process_manager
    
    def initialize_process(self, name, description):
        # Initialize a new process
        return process_id
    
    def execute_stage(self, stage, **kwargs):
        # Execute a specific stage
        return success
```

## Configuration

Edit `src/config.py` to configure your project:

```python
CONFIG = {
    # Data source (CSV or DB)
    'use_db': False,
    'db_url': "sqlite:///data/production.db",
    
    # Available algorithms
    'available_algorithms': ['example_algorithm'],
    
    # Process stages and decision points
    'stages': {
        'data_loading': {
            'sequence': 1,
            'decisions': { ... }
        },
        'data_transformation': { ... },
        'processing': { ... },
        'result_analysis': { ... }
    }
}
```

## License

[Your License]