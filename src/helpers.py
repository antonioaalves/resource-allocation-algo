"""File containing the helper functions for this project"""

# Import dependencies
import csv
import datetime
import logging
import os
from pandas import DataFrame
from typing import Dict, Any

# Local stuff
from src.config import PROJECT_NAME

# Get logger
logger = logging.getLogger(PROJECT_NAME)

def read_csv(file_path):
    """Function that reads a csv"""
    logger.info("Entering read_csv() function.")
    data = []
    try:
        logger.info(f"Reading CSV file from path {file_path}")
        with open(f"data/{file_path}", mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} does not exist.")
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        raise
    return data

def parse_bags(data: DataFrame) -> list:
    """
    Parse bags data from DataFrame into list of dictionaries.
    
    Args:
        data (DataFrame): DataFrame containing bag data with 'capacity' and 'color' columns
        
    Returns:
        list: List of dictionaries with bag information
    """
    logger.info("Entering parse_bags() function.")
    bags = []
    
    # Iterate over DataFrame rows using iterrows()
    for index, row in data.iterrows():
        try:
            # Convert row to dictionary and ensure capacity is float
            bag = {
                'capacity': float(row['capacity']),
                'color': row['color']
            }
            bags.append(bag)
        except ValueError as e:
            print(f"Invalid data in row {index}: {row}")
            logger.info(f"Invalid data in row {index}: {row}")
            logger.error(f"Error converting data: {str(e)}")
            
    logger.info(f"Returning bags with {len(bags)} items.")
    return bags


def parse_balls(data: DataFrame) -> list:
    """
    Parse balls data from DataFrame into list of dictionaries.
    
    Args:
        data (DataFrame): DataFrame containing ball data with 'capacity_contribution', 'colors', and 'id' columns
        
    Returns:
        list: List of dictionaries with ball information
    """
    logger.info("Entering parse_balls() function.")
    balls = []
    
    # Iterate over DataFrame rows using iterrows()
    for index, row in data.iterrows():
        try:
            # Check if colors is a string before processing
            colors_list = []
            if isinstance(row.get('colors'), str):
                colors_list = [color.strip() for color in row['colors'].split(',') if color.strip()]
            elif row.get('PRODUCTION_LINE_NAME') and isinstance(row['PRODUCTION_LINE_NAME'], str):
                # Try to use PRODUCTION_LINE_NAME as a fallback
                colors_list = [color.strip() for color in row['PRODUCTION_LINE_NAME'].split(',') if color.strip()]
            
            ball = {
                'id': row['ID'],
                'capacity_contribution': float(row['capacity_contribution']),
                'colors': colors_list
            }
            balls.append(ball)
        except ValueError as e:
            print(f"Invalid data in row {index}: {row}")
            logger.info(f"Invalid data in row {index}: {row}")
            logger.error(f"Error converting data: {str(e)}")
        except AttributeError as e:
            print(f"Invalid colors data in row {index}: {row.get('colors', 'colors not found')}")
            logger.error(f"Error processing colors: {str(e)}")
    
    logger.info(f"Returning balls with {len(balls)} items.")
    return balls

def save_output(allocations):
    filepath = "data/output/" + datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%OS')
    logger.info(f"Writing output to text file: {filepath}")

    # Create an data/output directory if it does not exist
    if not os.path.exists('data/output'):
        logger.info("Creating new directory for outputs...")
        os.makedirs('data/output')

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as txtfile:
            txtfile.write("BAG_ID | COLOR | OBJECT_CAPACITY | FILLED_CAPACITY | BALLS \n")
            txtfile.writelines(f"{bag_id} | ({allocation['color']}) | {allocation['objective_capacity']} | {allocation['filled_capacity']} | {allocation['balls']} \n" for bag_id, allocation in allocations.items())
    
    except IOError as e:
        logger.error(f"Unexpected error while writing to CSV file: {str(e)}")
        raise IOError(f"Error writing to CSV file: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Data structure doesn't match fieldnames: {str(e)}")
        raise ValueError(f"Data structure doesn't match fieldnames: {str(e)}")
    

def parse_allocations(allocation_str: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse allocation string in format "item:target:quantity,item:target:quantity,..."
    
    Example: "AE150:15:5,AE180:16:10" becomes 
    {'AE150': {'target': 15, 'quantity': 5}, 'AE180': {'target': 16, 'quantity': 10}}
    """
    # Handle empty string case
    if not allocation_str:
        return {}
    
    result = {}
    
    # Split by comma to get individual allocations
    allocations = allocation_str.split(',')
    
    for allocation in allocations:
        # Split each allocation by colon
        parts = allocation.split(':')
        
        # Check if the allocation has the expected format
        if len(parts) == 4:
            item, target, operating_type_id, quantity = parts
            result[item.strip()] = {
                'target': int(target.strip()),  # Convert to integer
                'operating_type_id': int(operating_type_id.strip()),
                'quantity': float(quantity.strip())
            }
        else:
            # Handle malformed input (optional)
            logger.warning(f"Warning: Skipping malformed allocation '{allocation}'")

    print(f"result: {result}")
    
    return result

def reconfigure_all_loggers(main_logger):
    """Force all existing loggers to use proper configuration"""
    # Get all existing loggers
    for name in logging.root.manager.loggerDict:
        if name == 'resource_allocation_algo':
            continue  # Skip your main logger to avoid duplicate handlers
            
        existing_logger = logging.getLogger(name)
        
        # Set them to DEBUG level
        existing_logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        for handler in existing_logger.handlers[:]:
            existing_logger.removeHandler(handler)
            
        # Copy handlers from your project logger
        for handler in main_logger.handlers:
            existing_logger.addHandler(handler)
            
    # Also fix the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)