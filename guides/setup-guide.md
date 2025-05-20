# Project Setup Guide

## Prerequisites

Before starting, ensure you have Python 3.8 or higher installed on your system. You can check your Python version by running:
```bash
python --version
```

## Step-by-Step Setup

### 1. Clone/Download the Project
- Download or clone this project to your local machine
- Navigate to the project directory in your terminal/command prompt

### 2. Set Up Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages (from the source directory)
pip install -r requirements.txt
```

### 4. Database Setup
```bash
# Create initial directory structure (if not done yet)
mkdir -p src/database
touch src/database/__init__.py

# Initialize Alembic
alembic init src/database/migrations

# Create initial migration
alembic revision --autogenerate -m "Initial database setup"

# Apply migration
alembic upgrade head
```

### 5. Directory Structure
Ensure you have the following directory structure:
```
project_root/
│
├── src/
│   ├── database/
│   │   ├── models.py
│   │   └── migrations/
│   ├── models.py
│   ├── helpers.py
│   ├── log_config.py
│   └── config.py
│
├── data/
│   ├── output/
│   └── *.csv files
│
├── logs/
├── alembic.ini
├── main.py
└── requirements.txt
```

### 6. Verify Setup
```python
# Run the main script to test everything is working
python main.py
```

## Common Issues & Solutions

### Issue: Missing Directories
If you encounter errors about missing directories, ensure all required directories exist:
```bash
mkdir -p data/output logs
```

### Issue: Database Errors
If you encounter database-related errors:
1. Delete the `production.db` file if it exists
2. Run the migration commands again:
```bash
alembic upgrade head
```

### Issue: Import Errors
If you see import errors:
1. Ensure you're in the project root directory
2. Verify your virtual environment is activated
3. Confirm all requirements are installed:
```bash
pip install -r requirements.txt
```

## Using the Application

### CSV Mode
To use the application with CSV files:
```python
from src.models import DataManagement

data_mgr = DataManagement(use_db=False)
data_mgr.load_csvs()
```

### Database Mode
To use the application with the database:
```python
from src.models import DataManagement

data_mgr = DataManagement(use_db=True)
data_mgr.load_data()
```

## Maintenance

### Adding New Models
When you add or modify database models:
1. Update the models in `src/database/models.py`
2. Create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```
3. Apply the migration:
```bash
alembic upgrade head
```

## Need Help?
If you encounter any issues not covered in this guide:
1. Check the application logs in the `logs/` directory
2. Verify all dependencies are correctly installed
3. Ensure your Python version is compatible
4. Contact the project maintainers for support