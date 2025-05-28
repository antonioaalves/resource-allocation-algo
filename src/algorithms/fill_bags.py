"""File containing the FillTheBags class"""

# Dependencies
import logging
import pandas as pd
import heapq
from base_data_project.algorithms.base import BaseAlgorithm
from base_data_project.storage.models import BaseDataModel

# Local stuff
from src.helpers import parse_bags, parse_balls
from src.config import PROJECT_NAME

# Get logger
logger = logging.getLogger(PROJECT_NAME)

class FillBagsAlgorithm(BaseAlgorithm):
    """Class containing the base FillTheBags logic"""

    def __init__(self, data: BaseDataModel, parameters=None, algo_name=None):
        # Use the passed algo_name or default to "FillTheBags"
        actual_algo_name = algo_name or "FillTheBags"
        # Initialize the parent first
        super().__init__(algo_name=actual_algo_name, parameters=parameters)

        logger.info("Initializing FillBagsAlgorithm object.")

        self.data = data
        self.status = ''
        self.filled = False
        self.bag_allocations = {}
        self.unused_balls = []
        self.unused_balls_capacity = 0        

    def adapt_data(self, adapted_data=None):
        """Transform the data for the algorithm - preserving IDs"""
        try:
            logger.info("Starting data transformation for FillTheBags")

            # Existing transformation logic
            balls_data = self.data.employee_df.copy()
            bags_allowed_for_bals = pd.merge(
                self.data.employee_production_lines_df,
                self.data.production_lines_table,
                left_on='PRODUCTION_LINE_ID',
                right_on='ID',
                how='left'
            ).groupby('EMPLOYEE_ID')['NAME'].agg(lambda x: ','.join(map(str, x))).reset_index()

            bags_data = self.data.production_lines_table.copy()
            
            # Log the input data
            logger.info(f"Input bags_data (prod lines): {len(bags_data)} rows, columns: {bags_data.columns.tolist()}")
            logger.info(f"Unique production line IDs: {bags_data['ID'].nunique()}")
            
            # Convert and transform data
            balls_data['ID'] = balls_data['ID'].astype(int)
            bags_allowed_for_bals['EMPLOYEE_ID'] = bags_allowed_for_bals['EMPLOYEE_ID'].astype(int)
            
            bags_allowed_for_bals = bags_allowed_for_bals.rename(columns={'NAME': 'PRODUCTION_LINE_NAME'})
            
            balls_data = pd.merge(
                balls_data,
                bags_allowed_for_bals,
                left_on='ID',
                right_on='EMPLOYEE_ID',
                how='left'
            )
            
            balls_data = balls_data.drop(['NAME', 'CONTRACT_TYPE', 'GROUP', 'TYPE', 'EMPLOYEE_ID'], axis=1)
            balls_data = balls_data.rename(columns={
                'PRODUCTION_LINE_NAME': 'colors',
                'CAPACITY_CONTRIBUTION': 'capacity_contribution'
            })
            
            # MODIFIED: Preserve the ID column in bags_data
            bags_data = bags_data.rename(columns={
                'NECESSITY': 'capacity',
                'NAME': 'color',
                'ID': 'bag_id'  # Keep ID as bag_id
            })
            
            # Log the transformed data
            logger.info(f"Transformed bags_data: {len(bags_data)} rows, columns: {bags_data.columns.tolist()}")
            logger.info(f"Unique bag IDs: {bags_data['bag_id'].nunique() if 'bag_id' in bags_data.columns else 'N/A'}")
            
            # Modify parse_bags to include bag_id
            def modified_parse_bags(data: pd.DataFrame) -> list:
                """Parse bags data from DataFrame into list of dictionaries."""
                logger.info("Entering modified parse_bags() function.")
                bags = []
                
                # Iterate over DataFrame rows using iterrows()
                for index, row in data.iterrows():
                    try:
                        # Convert row to dictionary and ensure capacity is float
                        bag = {
                            'capacity': float(row['capacity']),
                            'color': row['color'],
                            'bag_id': row['bag_id'] if 'bag_id' in row else index  # Include bag_id if available
                        }
                        bags.append(bag)
                    except ValueError as e:
                        print(f"Invalid data in row {index}: {row}")
                        logger.info(f"Invalid data in row {index}: {row}")
                        logger.error(f"Error converting data: {str(e)}")
                        
                logger.info(f"Returning bags with {len(bags)} items.")
                return bags
            
            self.bags = modified_parse_bags(bags_data)
            self.balls = parse_balls(balls_data)
            
            # Log the parsed data
            logger.info(f"Parsed bags: {len(self.bags)}")
            bag_ids = [bag.get('bag_id') for bag in self.bags]
            logger.info(f"Unique bag IDs: {len(set(bag_ids))}")
            logger.info(f"Parsed balls: {len(self.balls)}")
            
            logger.info("Data adaptation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data adaptation: {str(e)}")
            raise
        
    def execute_algorithm(self):
        """Execute the bag filling algorithm"""
        try:
            # Step 1: Initialize the data structures
            logger.info("Step 1: Initialize the data structures")
            remaining_capacity = {i: bag['capacity'] for i, bag in enumerate(self.bags)}
            bag_colors = {i: bag['color'] for i, bag in enumerate(self.bags)}
            bag_allocations = {
                i: {'color': bag_colors[i], 'balls': [], 'objective_capacity': self.bags[i]['capacity'], 'filled_capacity': 0}
                for i in range(len(self.bags))
            }
            used_balls = set()

            # Step 2: Separate balls by capacity contribution
            logger.info("Step 2: Separate balls by capacity contribution")
            high_capacity_balls = [ball for ball in self.balls if ball['capacity_contribution'] >= 1]
            low_capacity_balls = [ball for ball in self.balls if ball['capacity_contribution'] < 1]

            # Step 3: Sort balls by the number of colors (ascending)
            logger.info("Step 3: Sort balls by the number of colors (ascending)")
            high_capacity_balls.sort(key=lambda ball: len(ball['colors']))
            low_capacity_balls.sort(key=lambda ball: len(ball['colors']))

            # Combine balls lists, high capacity first
            sorted_balls = high_capacity_balls + low_capacity_balls

            # Step 4: Initialize a priority queue for bags based on remaining capacity
            logger.info("Step 4: Initialize a priority queue for bags based on remaining capacity")
            priority_queue = [(remaining_capacity[i], i) for i in range(len(self.bags))]
            heapq.heapify(priority_queue)

            # Step 5: Assign balls to bags
            logger.info("Step 5: Assign balls to bags")
            for i, ball in enumerate(sorted_balls):
                logger.info(f"Loop counter i: {i}/{(len(sorted_balls)-1)}")
                # Find all bags this ball can go into
                possible_bags = [i for i in range(len(self.bags)) if any(color in ball['colors'] for color in [bag_colors[i]])]

                # Sort possible bags by their remaining capacity
                possible_bags.sort(key=lambda bag: remaining_capacity[bag])

                # Try to place the ball in the bag with the smallest remaining capacity
                for j, bag in enumerate(possible_bags):
                    logger.info(f"Loop counter j: {j}/{(len(possible_bags)-1)}")
                    if remaining_capacity[bag] > 0:
                        remaining_capacity[bag] -= ball['capacity_contribution']
                        bag_allocations[bag]['balls'].append(ball['id'])
                        bag_allocations[bag]['filled_capacity'] += ball['capacity_contribution']
                        used_balls.add(ball['id'])

                        # Update the priority queue
                        heapq.heappush(priority_queue, (remaining_capacity[bag], bag))
                        logger.info("Breaking from j loop: remaining_capacity[bag] > 0")
                        break

            # Collect unused balls
            unused_balls = [ball for ball in self.balls if ball['id'] not in used_balls]
            unused_balls_ids = [ball['id'] for ball in unused_balls]
            unused_balls_capacity = sum(ball['capacity_contribution'] for ball in unused_balls)

            # Check if all bags are filled to their capacity
            all_filled = all(remaining_capacity[bag] <= 0 for bag in remaining_capacity)
            logger.info("Saving results in the algorithm result object")
            
            self.filled = all_filled
            self.bag_allocations = bag_allocations
            self.unused_balls_ids = unused_balls_ids
            self.unused_balls_capacity = unused_balls_capacity

            # Store detailed results
            # for bag_id, allocation in self.bag_allocations.items():

            logger.info("Bag filling completed successfully")

        except Exception as e:
            logger.error(f"Error during bag filling: {str(e)}")
            raise
            
    def format_results(self, algorithm_results=None):
        """
        Convert results and store them
        """
        pass

    def run(self, common_data):
        """Algorithm FillBags run method"""

        results = super().run(common_data)

        logger.info(f"Algorithm stage complete, data stored")

        return results