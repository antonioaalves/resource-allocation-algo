"""File containing the class responsible for all data operations"""

# Import dependencies
import pandas as pd
import numpy as np
from itertools import product
import logging
from typing import Dict, List, Any

# Local stuff
from src.config import CONFIG

logger = logging.getLogger('BagAllocationAlgo')

class AllocationData:
    """Container for data used in the allocation process"""
    
    def __init__(self):
        """Initialize empty data container"""
        # New dataframes
        self.contract_types_table = None
        self.days_number_table = None
        self.demands_table = None
        self.employees_table = None
        self.employee_hours_table = None
        self.employee_production_lines_table = None
        self.employee_shift_assignments_table = None
        self.groups_table = None
        self.line_types_table = None
        self.operating_type = None
        self.products_table = None
        self.production_lines_table = None
        self.production_lines_stats_table = None
        self.production_line_operating_type_table = None
        self.product_production_line_table = None
        self.sections_table = None
        self.shifts_table = None
        self.shift_types_table = None

        # Important dataframes not input (for later use)
        self.employee_df = None
        self.employee_hours_df = None
        self.employee_groups_df = None
        self.employee_production_lines_df = None
        self.product_production_line_df = None
        self.product_production_line_assignments_table = None
        self.product_production_line_assignments_df = None
        self.demands_df = None
        self.product_production_agg_df = None
    
    def load_from_data_manager(self, data_manager):
        """Load all required data using the data manager"""
        try:
            logger.info("Loading data from data manager")

            # TODO: new data tables (remove the previous)
            self.contract_types_table = data_manager.load_data('contract_types')
            self.demands_table = data_manager.load_data('demands')
            self.days_number_table = data_manager.load_data('days_numbers')
            self.employees_table = data_manager.load_data('employees')
            self.employee_hours_table = data_manager.load_data('employee_hours')
            self.employee_production_lines_table = data_manager.load_data('employee_production_lines')
            self.employee_shift_assignments_table = data_manager.load_data('employee_shift_assignments')
            self.groups_table = data_manager.load_data('groups')
            self.line_types_table = data_manager.load_data('line_types')
            self.operating_types_table = data_manager.load_data('operating_types')
            self.products_table = data_manager.load_data('products')
            self.production_lines_table = data_manager.load_data('production_lines')
            self.production_lines_stats_table = data_manager.load_data('production_lines_stats')
            self.production_line_operating_type_table = data_manager.load_data('production_lines_operating_types')
            self.product_production_line_assignments_table = data_manager.load_data('product_production_line_assignments')
            self.product_production_line_table = data_manager.load_data('product_production_lines')
            self.sections_table = data_manager.load_data('sections')
            self.shifts_table = data_manager.load_data('shifts')
            self.shift_types_table = data_manager.load_data('shift_types')
                
            return True
        except Exception as e:
            logger.error(f"Error loading data from data manager: {str(e)}")
            return False
    
    def validate(self):
        """Validate that the required data is present and valid"""
        missing_data = []
        empty_df = []
        empty_tables = CONFIG.get('empty_dataframes', ['employee_shift_assignments_table', 'product_production_line_assignments_table']) # If any tables could be empty
        
        # Get all attributes that end with '_table'
        table_attributes = [attr for attr in dir(self) if attr.endswith('_table')]
        
        # Check each attribute
        for attr in table_attributes:
            df = getattr(self, attr)
            if df is None:
                missing_data.append(attr)

            if attr in empty_tables:
                continue
            elif len(df) == 0:
                empty_df.append(attr)
            

        # Check if all required dataframes were loaded
        if len(missing_data) > 0:
            logger.error(f"One or more required dataframes failed to load: {attr}")
            return False
        
        # Check if dataframes have data
        if len(empty_df) > 0:
            logger.error(f"One or more required dataframes is empty: {empty_df}")
            return False
        
        # Check for required columns in each dataframe
        # TODO: Add it to config file or any other file containing validations needed for each dataframe
        
        logger.info("Data validation passed")
        return True
    
    def filter_by_decision_time(self, months, years):
        """Filter method to only load data for months and years defined"""

        try:
            months = [int(m) if isinstance(m, str) else m for m in months]
            years = [int(y) if isinstance(y, str) else y for y in years]
            # Add 'month' to the dataframes that have only date
            self.shifts_table['day'] = pd.to_datetime(self.shifts_table['day'], dayfirst=True)
            self.shifts_table['month'] = self.shifts_table['day'].dt.month
            self.shifts_table['year'] = self.shifts_table['day'].dt.year

            # Filter by months and years
            self.demands_table = self.demands_table[self.demands_table['month'].isin(months)]
            self.demands_table = self.demands_table[self.demands_table['year'].isin(years)]

            self.employee_hours_table = self.employee_hours_table[self.employee_hours_table['month'].isin(months)]
            self.employee_hours_table = self.employee_hours_table[self.employee_hours_table['year'].isin(years)]

            self.production_lines_stats_table = self.production_lines_stats_table[self.production_lines_stats_table['month'].isin(months)]
            self.production_lines_stats_table = self.production_lines_stats_table[self.production_lines_stats_table['year'].isin(years)]

            self.shifts_table = self.shifts_table[self.shifts_table['month'].isin(months)]
            self.shifts_table = self.shifts_table[self.shifts_table['year'].isin(years)]
            
            return True

        except Exception as e:
            logger.error(f"Error filtering tables by month and year decisions: {e}")
            return False
        
    def join_dataframes(self, months, years):
        """Method for joining the existing dataframes"""

        try:
            # Save dataframes in variables
            employee_df = self.employees_table.copy()
            employee_hours_df = self.employee_hours_table.copy()
            shift_df = self.shifts_table.copy()
            employee_production_lines_df = self.employee_production_lines_table.copy()
            demands_df = self.demands_table.copy()
            production_line_operating_type_df = self.production_line_operating_type_table.copy()

            # ---------------------
            # EMPLOYEE INFORMATION
            # ---------------------
            
            employee_df = self.employees_table.merge(self.contract_types_table, how='inner', left_on='contract_type_id', right_on='id')
            employee_df = employee_df.rename(columns={
                'name_y': 'contract_type',
                'name_x': 'name', 
                'id_x': 'id'
            })
            employee_df = employee_df.drop(columns=['id_y'])
            # Groups data
            employee_df = employee_df.merge(self.groups_table, how='inner', left_on='group_id', right_on='id')
            employee_df = employee_df.rename(columns={
                'name_y': 'group_name',
                'name_x': 'name', 
                'id_x': 'id'
            })
            employee_df = employee_df.drop(columns=['id_y'])
            # Sections data
            employee_df = employee_df.merge(self.sections_table, how='inner', left_on='section_id', right_on='id')
            employee_df = employee_df.rename(columns={
                'name_y': 'section_name',
                'name_x': 'name', 
                'id_x': 'id'
            })
            employee_df = employee_df.drop(columns=['id_y'])
            employee_df['group_id'] = employee_df['group_id'].astype(int)

            # ---------------------
            # SHIFT INFORMATION
            # ---------------------
             
            shift_df = shift_df.merge(self.shift_types_table, how='inner', left_on='shifttype_id', right_on='id')
            shift_df = shift_df.rename(columns={
                'id_x': 'id'
            })
            shift_df = shift_df.drop(columns=['name', 'id_y', 'shifttype_id'])
            shift_df = shift_df.merge(self.groups_table, how='inner', left_on='group_id', right_on='id')
            shift_df = shift_df.rename(columns={
                'id_x': 'id',
                'name_y': 'group_name',
                'name_x': 'name'
            })
            shift_df = shift_df.drop(columns=['id_y'])
            employee_groups_df = employee_df.drop(columns=['name', 'contract_type', 'employee_type', 'capacity_contribution', 'bank_hours', 'section_id', 'contract_type', 'section_name', 'group_name'])
            employee_groups_df = employee_groups_df.rename(columns={'id': 'employee_id'})
            employee_groups_df = employee_groups_df.merge(shift_df, how='inner', left_on='group_id', right_on='group_id')
            employee_groups_df = employee_groups_df.drop(columns=['id'])

            # ------------------------------------
            # PRODUCT/PRODUCTION LINES INFORMATION
            # ------------------------------------

            product_production_line_df = self.product_production_line_table.merge(self.products_table, how='inner', left_on='product_id', right_on='id')
            product_production_line_df = product_production_line_df.rename(columns={
                'id_x': 'id',
                'name': 'product_name',
                'diameter': 'product_diameter',
                'height': 'product_height',
                'material': 'product_material'
            })
            product_production_line_df = product_production_line_df.drop(columns=['id_y'])
            product_production_line_df = product_production_line_df.merge(self.production_lines_table, how='inner', left_on='production_line_id', right_on='id')
            product_production_line_df = product_production_line_df.rename(columns={
                'id_x': 'id',
                'name': 'production_line_name'
            })
            product_production_line_df = product_production_line_df.drop(columns=['id_y'])
            product_production_line_df = product_production_line_df.merge(self.production_lines_stats_table, how='inner', left_on='id', right_on='product_production_line_id')
            product_production_line_df = product_production_line_df.rename(columns={
                'id_x': 'id',
                'name': 'production_line_name'
            })
            product_production_line_df = product_production_line_df.drop(columns=['id_y'])
            product_production_line_df = product_production_line_df.drop(columns=['product_production_line_id'])

            # ----------------------------------------------------
            # PRODUCT/PRODUCTION LINES/OPERATING TYPES INFORMATION
            # ----------------------------------------------------

            # Add calculations to determine the various values for different production lines based on the operating types
            # Initialize a dict to store the values for the amount of possible shifts each month
            shifts_amount = {
                'id': [i for i in range(1, 2*len(months) + 1)],
                'operating_type_id': [i for i in range(1,3) for _ in range(len(months))], # range(1,3) is to repeat 1 and 2 len(months) times
                'month': months * 2,                                                      # *2: because we have two shifts
                'year': years * len(months) * 2                                           # *2: because we have two shifts
            }

            shifts_amount_df = pd.DataFrame(shifts_amount)

            merged_df = pd.merge(
                shifts_amount_df,
                self.days_number_table,
                on=['month', 'year'],
                how='left',
                suffixes=('', '_days')
            )

            merged_df['shifts_amount'] = np.where(
                merged_df['operating_type_id'] == 1,
                merged_df['working_days'] * 3,
                merged_df['working_days'] * 3 + merged_df['saturdays'] * 3 + merged_df['sundays'] * 2
            )
            shifts_amount_df = pd.DataFrame(merged_df[['id', 'operating_type_id', 'month', 'year', 'shifts_amount']])

            prodlines = list(set((production_line_operating_type_df['production_line_id'].to_list())))
            max_id = production_line_operating_type_df['id'].max()
            
            # 1. Get the unique production lines, years, and months we're working with
            production_lines = production_line_operating_type_df['production_line_id'].unique()
            target_months = months  # [1, 2, 3]
            target_years = years     # [2024]

            # 2. Create a DataFrame with all combinations we need to ensure exist
            all_combinations = pd.DataFrame(list(product(
                production_lines, 
                [1, 2, 3],  # operating_type_ids
                target_months,
                target_years
            )), columns=['production_line_id', 'operating_type_id', 'month', 'year'])

            # 3. Filter for just the operating_type_id 1 and 2 combinations that we need to calculate
            missing_combinations = all_combinations[all_combinations['operating_type_id'].isin([1, 2])]

            # 4. Create a mapping from type 3 data for stoppage_shifts
            type3_data = production_line_operating_type_df[
                (production_line_operating_type_df['operating_type_id'] == 3)
            ].copy().rename(columns={'number_of_shifts': 'stoppage_shifts'})

            # 5. Merge to get stoppage_shifts for each production line, month, year
            result = missing_combinations.merge(
                type3_data[['production_line_id', 'month', 'year', 'stoppage_shifts']],
                on=['production_line_id', 'month', 'year'],
                how='left'
            )

            # 6. Create a mapping from shifts_amount_df
            shifts_data = shifts_amount_df[
                (shifts_amount_df['operating_type_id'].isin([1, 2])) &
                (shifts_amount_df['month'].isin(target_months)) &
                (shifts_amount_df['year'].isin(target_years))
            ].copy()[['operating_type_id', 'month', 'year', 'shifts_amount']]

            # 7. Merge to get shifts_amount based on operating_type_id, month, year
            result = result.merge(
                shifts_data,
                on=['operating_type_id', 'month', 'year'],
                how='left'
            )

            # 8. Calculate the number_of_shifts as shifts_amount - stoppage_shifts
            result['number_of_shifts'] = result['shifts_amount'] - result['stoppage_shifts']

            # 9. Clean up and prepare for merging
            result.drop(['stoppage_shifts', 'shifts_amount'], axis=1, inplace=True)

            # 10. Generate new IDs
            next_id = max_id + 1
            result['id'] = range(next_id, next_id + len(result))
            max_id = next_id + len(result) - 1

            # 11. Get the combinations that aren't already in the DataFrame
            existing_combos = set(map(tuple, production_line_operating_type_df[
                ['production_line_id', 'operating_type_id', 'month', 'year']
            ].values))

            result = result[~result.apply(
                lambda row: (row['production_line_id'], row['operating_type_id'], row['month'], row['year']) in existing_combos, 
                axis=1
            )]

            # 12. Append only the new combinations to the original DataFrame
            production_line_operating_type_df = pd.concat([production_line_operating_type_df, result], ignore_index=True)

            # Store the df in another variable
            aux_df = production_line_operating_type_df.copy()

            aux_df['theoretical_hours_amount'] = np.where(
                aux_df['operating_type_id'] == 3,
                0,
                aux_df['number_of_shifts'].apply(lambda x: x*8)
            )
            aux_df = pd.DataFrame(aux_df[['id', 'production_line_id', 'operating_type_id', 'month', 'year', 'number_of_shifts', 'theoretical_hours_amount']])
            production_line_operating_type_df = aux_df

            # Merge data to product production line
            merged_df = pd.merge(
                product_production_line_df,
                demands_df,
                on=['product_id', 'month', 'year'],
                how='left',
                suffixes=('', '_from_df2')
            )
            merged_df = merged_df.drop(columns=['id_from_df2'])
            merged_df = merged_df.rename(columns={'value': 'demand'})
            product_production_line_df = merged_df.copy()

            # Step 1: Calculate needed_hours in the original dataframe
            aux_df = product_production_line_df.copy()
            aux_df['needed_hours'] = (aux_df['demand'] * 1000) / (aux_df['production_rate'] * aux_df['oee'])

            # Step 2: Create a dataframe with the sum of needed_hours for the specific combination you need
            # This will ensure real_hours_amount is the same for each month-year-production_line-diameter-section combination
            aggregation_df = aux_df.groupby(['production_line_id', 'product_diameter', 'section_id', 'month', 'year'])['needed_hours'].sum().reset_index()
            aggregation_df = aggregation_df.rename(columns={'needed_hours': 'real_hours_amount'})

            # Step 3: Merge this back to the original data
            result = aux_df.merge(
                aggregation_df,
                on=['production_line_id', 'product_diameter', 'section_id', 'month', 'year'],
                how='left'
            )

            # Step 4: Merge with operating type info
            result2 = result.merge(
                production_line_operating_type_df[['production_line_id', 'operating_type_id', 'month', 'year', 'theoretical_hours_amount']],
                on=['production_line_id', 'month', 'year'],
                how='left'
            )

            # Step 5: Handle NaN values that might occur after the merge
            result2['theoretical_hours_amount'] = result2['theoretical_hours_amount'].fillna(0)
            result2['operating_type_id'] = result2['operating_type_id'].fillna(0)

            # Step 6: Create the validity flag
            result2['valid_operating_type'] = (
                (result2['real_hours_amount'] <= result2['theoretical_hours_amount']) | 
                (result2['operating_type_id'] == 3)
            )

            # Logs (TODO: add them)
            #product_count_original = aux_df['product_id'].nunique()
            #product_count_result = result2['product_id'].nunique()
            #print(f"Number of unique products in original data: {product_count_original}")
            #print(f"Number of unique products in result: {product_count_result}")
            ## Verify that real_hours_amount is consistent across the specified combinations
            #check_consistency = result2.groupby(['production_line_id', 'product_diameter', 'section_id', 'month', 'year'])['real_hours_amount'].nunique()
            #print("\nNumber of unique real_hours_amount values per combination:")
            #print(check_consistency.value_counts())

            result2['delta_hours_amount'] = np.where(
                (result2['operating_type_id'] == 3),
                0,
                result2['theoretical_hours_amount'] - result2['real_hours_amount']
            )

            product_production_agg_df = result2.copy()

            # ------------------------------------
            # FINAL TOUCHES
            # ------------------------------------

            # Drop NA's (TODO: create a way to register which id's were droped)
            employee_df.dropna()
            employee_groups_df.dropna()
            product_production_line_df.dropna()

            # Add a total to employee_hours_table
            employee_hours_df['total_hours_leave'] = employee_hours_df['leave_fte'] + employee_hours_df['loan_fte'] + employee_hours_df['holidays_fte'] + employee_hours_df['training_hours']

            # ------------------------------------
            # ASSIGNING TO CLASS ATTRIBUTES
            # ------------------------------------

            self.employee_df = employee_df.copy()
            self.employee_hours_df = employee_hours_df.copy()
            self.employee_groups_df = employee_groups_df.copy()
            self.employee_production_lines_df = employee_production_lines_df.copy()
            self.product_production_line_df = product_production_line_df.copy()
            self.demands_df = demands_df.copy()
            self.product_production_agg_df = product_production_agg_df.copy()

            # --------------------------------------------------------
            # CREATE PRINTABLE DATAFRAME FOR PRODUCT ALLOCATION STAGE
            # --------------------------------------------------------

            self.printable_df = self.product_production_agg_df.copy()
            self.printable_df = self.printable_df[['product_id', 'production_line_id', 'product_name', 'production_line_name', 'month', 'year', 'real_hours_amount', 'operating_type_id', 'theoretical_hours_amount', 'delta_hours_amount']]

            # Log this activities (TODO: see if there is no other information to be saved in logs)
            logger.info(f"Employee df columns: {self.employee_df.columns}")
            logger.info(f"Employee hours df columns: {self.employee_hours_df.columns}")
            logger.info(f"Employee shifts df columns: {self.employee_groups_df.columns}")
            logger.info(f"Employee production lines df columns: {self.employee_production_lines_df.columns}")
            logger.info(f"Product production lines df columns: {self.product_production_line_df.columns}")
            logger.info(f"Demand df columns: {self.demands_df.columns}")

            return True
        
        except Exception as e:
            logger.error(f"Error joining/merging dataframes: {e}")
            return False

    def assign_products(self, product_assignments: Dict[str, List[int]]):
        """
        Method for assigning produts to production lines.
        Inputs:
            - products_assignments: dictionary containing assignments by product_id, production_line_id, quantity
        """

        try:
            # TODO: Add validation, to only allow users to do valid additions addition  
            logger.info("Starting assigning products according to user decisions")

            # Create a list to hold all the rows
            rows = []

            # Convert each product and its data into a row
            for product_id, values in product_assignments.items():
                row = {
                    'PRODUCT_ID': product_id,
                    'PRODUCTION_LINE': values['target'],
                    'OPERATING_TYPE_ID': values['operating_type_id'],
                    'QUANTITY': values['quantity']
                }
                rows.append(row)

            # Create DataFrame from the list of rows
            df = pd.DataFrame(rows)

            # Convert columns to appropriate data types
            df['PRODUCT_ID'] = df['PRODUCT_ID'].astype(int)
            df['PRODUCTION_LINE'] = df['PRODUCTION_LINE'].astype(int)
            df['OPERATING_TYPE_ID'] = df['OPERATING_TYPE_ID'].astype(int)
            df['QUANTITY'] = df['QUANTITY'].astype(float)

            self.product_production_line_assignments_df = df.copy()
            print(f"Assignments: {self.product_production_line_assignments_df}")

            # TODO: Remove this line:
            self.product_production_line_assignments_df.to_csv('C:/Users/antonio.alves/Documents/personal-stuff/projects/alocation-algo/operational-flexibility/data/output/product_assignments.csv')
            return True
        except Exception as e:
            logger.error(f"Error assigning products according to user decisions: {e}")
            return False
        
    def validate_product_assignments(self, product_assignments: Dict[str, Dict[str, Any]]):
        """
        Method for validating product assignments
        Inputs:
            - product_assignments: dictionary containing the product allocations from user decisions. (output from parse_allocations)
        """
        
        try:
            unexisting_products = []
            unexisting_prodlines = []
            unexisting_operating_types = []
            unexisting_combinations = []
            invalid_quantities = []
            
            # Convert unique values to appropriate types if needed
            unique_products = self.product_production_agg_df['product_id'].unique()
            unique_prodlines = self.product_production_agg_df['production_line_id'].unique()
            unique_operating_types = self.product_production_agg_df['operating_type_id'].unique()
            unique_product_prodline_comb = self.product_production_line_df[['product_id', 'production_line_id']].drop_duplicates()
            
            # Check products
            for item_name, item_data in product_assignments.items():
                # Access as dictionary since that's what we have
                item_name = int(item_name)
                target = item_data['target']
                operating_type_id = item_data['operating_type_id']
                quantity = item_data['quantity']
                # Make sure we're comparing compatible types
                if item_name not in unique_products:
                    unexisting_products.append(item_name)
                
                if isinstance(target, str):
                    try:
                        target = int(target)
                    except ValueError:
                        pass  # Keep as string if can't convert
                        
                if target not in unique_prodlines:
                    unexisting_prodlines.append(target)
                    
                if operating_type_id not in unique_operating_types:
                    unexisting_operating_types.append(item_data[1])
                    
                if quantity <= 0:
                    invalid_quantities.append(item_data[2])

                # Handle type consistency for comparison
                matches_combination = unique_product_prodline_comb[
                    (unique_product_prodline_comb['product_id'] == item_name) & 
                    (unique_product_prodline_comb['production_line_id'] == target)
                ]
                
                if len(matches_combination) != 1: 
                    unexisting_combinations.append(f"product:{item_name}-prod_line:{target}")

            if (len(unexisting_products) > 0 or 
                len(unexisting_prodlines) > 0 or 
                len(unexisting_operating_types) > 0 or 
                len(unexisting_combinations) > 0):
                logger.warning(f"Invalid assignments. Products: {unexisting_products}, "
                            f"production_lines: {unexisting_prodlines}, "
                            f"operating_types: {unexisting_operating_types}, "
                            f"product-prodlines combinations: {unexisting_combinations}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating product assignments according to user decisions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def calculate_capacity_contributions(self):
        """
        Method for calculating each employee capacity contribution based on the other dataframe information
        """

        try:
            # Logic for determining the capacity contribution

            return True
        except Exception as e:
            logger.error(f"")
            return False
        
    def calculate_capacity_contributions(self):
        """
        Method for calculating each employee capacity contribution based on the other dataframe information
        """

        try:
            # Logic for determining the capacity contribution

            return True
        except Exception as e:
            logger.error(f"")
            return False