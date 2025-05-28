"""File containing the class responsible for all data operations"""

# Import dependencies
import pandas as pd
import numpy as np
from itertools import product
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from base_data_project.data_manager.managers import BaseDataManager
from base_data_project.storage.models import BaseDataModel
from base_data_project.storage.containers import BaseDataContainer

# Local stuff
from src.config import CONFIG, PROJECT_NAME

logger = logging.getLogger(PROJECT_NAME)

class AllocationData(BaseDataModel):
    """Container for data used in the allocation process"""
    
    def __init__(self, data_container: Optional[BaseDataContainer] = None, project_name: str = 'base_data_project'):
        """Initialize data container"""
        # Super init with data container
        super().__init__(data_container=data_container, project_name=project_name)
        
        # Initialize process ID for storage
        self.process_id = None
        
        # Track original data loading status
        self._data_loaded = False
        self._data_transformed = False
        self._data_filtered = False
        
        logger.info("AllocationData initialized with data container")

    def load_from_data_manager(self, data_manager: BaseDataManager):
        """Load all required data using the data manager"""
        try:
            logger.info("Loading data from data manager")

            # Load all required tables
            tables = {
                'contract_types': data_manager.load_data('contract_types'),
                'demands': data_manager.load_data('demands'),
                'days_number': data_manager.load_data('days_numbers'),
                'employees': data_manager.load_data('employees'),
                'employee_hours': data_manager.load_data('employee_hours'),
                'employee_production_lines': data_manager.load_data('employee_production_lines'),
                'employee_shift_assignments': data_manager.load_data('employee_shift_assignments'),
                'groups': data_manager.load_data('groups'),
                'line_types': data_manager.load_data('line_types'),
                'operating_types': data_manager.load_data('operating_types'),
                'products': data_manager.load_data('products'),
                'production_lines': data_manager.load_data('production_lines'),
                'production_lines_stats': data_manager.load_data('production_lines_stats'),
                'production_line_operating_type': data_manager.load_data('production_lines_operating_types'),
                'product_production_line_assignments': data_manager.load_data('product_production_line_assignments'),
                'product_production_line': data_manager.load_data('product_production_lines'),
                'sections': data_manager.load_data('sections'),
                'shifts': data_manager.load_data('shifts'),
                'shift_types': data_manager.load_data('shift_types')
            }
            
            # Store all tables in the data container
            if self.data_container:
                # Store the loaded data in the data container
                storage_key = self.data_container.store_stage_data(
                    stage_name="data_loading",
                    data=tables,
                    metadata={
                        "process_id": self.process_id or "default_process",
                        "data_type": "tables",
                        "table_count": len(tables),
                        "tables": list(tables.keys())
                    }
                )
                logger.info(f"Stored loaded data in container with key: {storage_key}")
            
            # For backward compatibility and direct access convenience,
            # also store references to tables as attributes
            for table_name, table_data in tables.items():
                setattr(self, f"{table_name}_table", table_data)
            
            self._data_loaded = True
            logger.info("Data loading completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data from data manager: {str(e)}")
            return False
    
    def validate(self):
        """Validate that the required data is present and valid"""
        # If using data container, retrieve the latest data
        if self.data_container and self._data_loaded:
            try:
                tables = self.data_container.retrieve_stage_data(
                    stage_name="data_loading",
                    process_id=self.process_id or "default_process"
                )
            except Exception as e:
                logger.error(f"Error retrieving loaded data: {str(e)}")
                # Fall back to instance attributes
                tables = {}
        else:
            # Use instance attributes if no data container
            tables = {}
        
        missing_data = []
        empty_df = []
        empty_tables = CONFIG.get('empty_dataframes', 
                                ['employee_shift_assignments_table', 
                                 'product_production_line_assignments_table'])
        
        # Check each table
        for table_name in [
            'contract_types', 'demands', 'days_number', 'employees',
            'employee_hours', 'employee_production_lines', 'employee_shift_assignments',
            'groups', 'line_types', 'operating_types', 'products',
            'production_lines', 'production_lines_stats', 'production_line_operating_type',
            'product_production_line_assignments', 'product_production_line',
            'sections', 'shifts', 'shift_types'
        ]:
            # Check if table exists in the data container
            if table_name in tables:
                df = tables[table_name]
            else:
                # Fall back to instance attribute
                attr_name = f"{table_name}_table"
                df = getattr(self, attr_name, None)
            
            if df is None:
                missing_data.append(table_name)
                continue
                
            if table_name not in empty_tables and len(df) == 0:
                empty_df.append(table_name)

        # Check if all required dataframes were loaded
        if len(missing_data) > 0:
            logger.error(f"One or more required dataframes failed to load: {missing_data}")
            return False
        
        # Check if dataframes have data
        if len(empty_df) > 0:
            logger.error(f"One or more required dataframes is empty: {empty_df}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def filter_by_decision_time(self, months, years):
        """Filter method to only load data for months and years defined"""
        try:
            # First, retrieve the latest data if using data container
            if self.data_container and self._data_loaded:
                try:
                    tables = self.data_container.retrieve_stage_data(
                        stage_name="data_loading",
                        process_id=self.process_id or "default_process"
                    )
                except Exception as e:
                    logger.error(f"Error retrieving loaded data: {str(e)}")
                    # Fall back to instance attributes
                    tables = {
                        attr_name.replace('_table', ''): getattr(self, attr_name)
                        for attr_name in dir(self)
                        if attr_name.endswith('_table') and not attr_name.startswith('_')
                    }
            else:
                # Use instance attributes if no data container
                tables = {
                    attr_name.replace('_table', ''): getattr(self, attr_name)
                    for attr_name in dir(self)
                    if attr_name.endswith('_table') and not attr_name.startswith('_')
                }
            
            # Ensure months and years are integers
            months = [int(m) if isinstance(m, str) else m for m in months]
            years = [int(y) if isinstance(y, str) else y for y in years]
            
            # Create filtered copies of the tables
            filtered_tables = {}
            
            # Copy tables that don't need filtering
            for table_name, df in tables.items():
                if table_name not in ['demands', 'employee_hours', 'production_lines_stats', 
                                     'shifts', 'production_line_operating_type']:
                    filtered_tables[table_name] = df.copy()
            
            # Add 'month' to shifts table
            shifts_df = tables['shifts'].copy()
            shifts_df['day'] = pd.to_datetime(shifts_df['day'], dayfirst=True)
            shifts_df['month'] = shifts_df['day'].dt.month
            shifts_df['year'] = shifts_df['day'].dt.year
            
            # Filter tables by months and years
            filtered_tables['demands'] = tables['demands'][
                tables['demands']['month'].isin(months) & 
                tables['demands']['year'].isin(years)
            ].copy()
            
            filtered_tables['employee_hours'] = tables['employee_hours'][
                tables['employee_hours']['month'].isin(months) & 
                tables['employee_hours']['year'].isin(years)
            ].copy()
            
            filtered_tables['production_lines_stats'] = tables['production_lines_stats'][
                tables['production_lines_stats']['month'].isin(months) & 
                tables['production_lines_stats']['year'].isin(years)
            ].copy()
            
            filtered_tables['shifts'] = shifts_df[
                shifts_df['month'].isin(months) & 
                shifts_df['year'].isin(years)
            ].copy()
            
            filtered_tables['production_line_operating_type'] = tables['production_line_operating_type'][
                tables['production_line_operating_type']['month'].isin(months) & 
                tables['production_line_operating_type']['year'].isin(years)
            ].copy()
            
            # Store filtered data in data container
            if self.data_container:
                storage_key = self.data_container.store_stage_data(
                    stage_name="data_filtering",
                    data=filtered_tables,
                    metadata={
                        "process_id": self.process_id or "default_process",
                        "filter_months": months,
                        "filter_years": years,
                        "data_type": "filtered_tables"
                    }
                )
                logger.info(f"Stored filtered data in container with key: {storage_key}")
            
            # For backward compatibility and direct access,
            # update instance attributes
            for table_name, df in filtered_tables.items():
                setattr(self, f"{table_name}_table", df)
            
            self._data_filtered = True
            logger.info(f"Successfully filtered data for months {months} and years {years}")
            return True
            
        except Exception as e:
            logger.error(f"Error filtering tables by month and year decisions: {e}")
            return False
        
    def join_dataframes(self, months, years):
        """Method for joining the existing dataframes"""
        try:
            # Retrieve the latest filtered data if available
            if self.data_container and self._data_filtered:
                try:
                    tables = self.data_container.retrieve_stage_data(
                        stage_name="data_filtering",
                        process_id=self.process_id or "default_process"
                    )
                except Exception as e:
                    logger.error(f"Error retrieving filtered data: {str(e)}")
                    # Fall back to instance attributes
                    tables = {
                        attr_name.replace('_table', ''): getattr(self, attr_name)
                        for attr_name in dir(self)
                        if attr_name.endswith('_table') and not attr_name.startswith('_')
                    }
            else:
                # Use instance attributes
                tables = {
                    attr_name.replace('_table', ''): getattr(self, attr_name)
                    for attr_name in dir(self)
                    if attr_name.endswith('_table') and not attr_name.startswith('_')
                }
            
            # Create joined dataframes - follows the original implementation
            # but uses the tables dictionary instead of attributes directly
            
            # Save dataframes in variables for cleaner code
            employee_df = tables['employees'].copy()
            employee_hours_df = tables['employee_hours'].copy()
            shift_df = tables['shifts'].copy()
            employee_production_lines_df = tables['employee_production_lines'].copy()
            demands_df = tables['demands'].copy()
            production_line_operating_type_df = tables['production_line_operating_type'].copy()

            # ---------------------
            # EMPLOYEE INFORMATION
            # ---------------------
            
            employee_df = employee_df.merge(tables['contract_types'], how='inner', left_on='contract_type_id', right_on='id')
            employee_df = employee_df.rename(columns={
                'name_y': 'contract_type',
                'name_x': 'name', 
                'id_x': 'id'
            })
            employee_df = employee_df.drop(columns=['id_y'])
            
            # Groups data
            employee_df = employee_df.merge(tables['groups'], how='inner', left_on='group_id', right_on='id')
            employee_df = employee_df.rename(columns={
                'name_y': 'group_name',
                'name_x': 'name', 
                'id_x': 'id'
            })
            employee_df = employee_df.drop(columns=['id_y'])
            
            # Sections data
            employee_df = employee_df.merge(tables['sections'], how='inner', left_on='section_id', right_on='id')
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
             
            shift_df = shift_df.merge(tables['shift_types'], how='inner', left_on='shifttype_id', right_on='id')
            shift_df = shift_df.rename(columns={'id_x': 'id'})
            shift_df = shift_df.drop(columns=['name', 'id_y', 'shifttype_id'])
            
            shift_df = shift_df.merge(tables['groups'], how='inner', left_on='group_id', right_on='id')
            shift_df = shift_df.rename(columns={
                'id_x': 'id',
                'name_y': 'group_name',
                'name_x': 'name'
            })
            shift_df = shift_df.drop(columns=['id_y'])
            
            employee_groups_df = employee_df.drop(columns=['name', 'contract_type', 'employee_type', 
                                                         'capacity_contribution', 'bank_hours', 'section_id', 
                                                         'contract_type', 'section_name', 'group_name'])
            employee_groups_df = employee_groups_df.rename(columns={'id': 'employee_id'})
            employee_groups_df = employee_groups_df.merge(shift_df, how='inner', left_on='group_id', right_on='group_id')
            employee_groups_df = employee_groups_df.drop(columns=['id'])

            # ------------------------------------
            # PRODUCT/PRODUCTION LINES INFORMATION
            # ------------------------------------

            product_production_line_df = tables['product_production_line'].merge(
                tables['products'], how='inner', left_on='product_id', right_on='id')
            product_production_line_df = product_production_line_df.rename(columns={
                'id_x': 'id',
                'name': 'product_name',
                'diameter': 'product_diameter',
                'height': 'product_height',
                'material': 'product_material'
            })
            product_production_line_df = product_production_line_df.drop(columns=['id_y'])
            
            product_production_line_df = product_production_line_df.merge(
                tables['production_lines'], how='inner', left_on='production_line_id', right_on='id')
            product_production_line_df = product_production_line_df.rename(columns={
                'id_x': 'id',
                'name': 'production_line_name'
            })
            product_production_line_df = product_production_line_df.drop(columns=['id_y'])
            
            product_production_line_df = product_production_line_df.merge(
                tables['production_lines_stats'], how='inner', left_on='id', right_on='product_production_line_id')
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
                'operating_type_id': [i for i in range(1,3) for _ in range(len(months))], 
                'month': months * 2,
                'year': years * len(months) * 2
            }

            shifts_amount_df = pd.DataFrame(shifts_amount)

            merged_df = pd.merge(
                shifts_amount_df,
                tables['days_number'],
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
            target_months = months
            target_years = years

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

            # Step 2: Create a dataframe with the sum of needed_hours for the specific combination
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

            result2['delta_hours_amount'] = np.where(
                (result2['operating_type_id'] == 3),
                0,
                result2['theoretical_hours_amount'] - result2['real_hours_amount']
            )

            product_production_agg_df = result2.copy()

            # ------------------------------------
            # FINAL TOUCHES
            # ------------------------------------

            # Drop NA's
            employee_df.dropna(inplace=True)
            employee_groups_df.dropna(inplace=True)
            product_production_line_df.dropna(inplace=True)

            # Add a total to employee_hours_table
            employee_hours_df['total_hours_leave'] = (
                employee_hours_df['leave_fte'] + 
                employee_hours_df['loan_fte'] + 
                employee_hours_df['holidays_fte'] + 
                employee_hours_df['training_hours']
            )

            # Create printable DataFrame for product allocation stage
            printable_df = product_production_agg_df.copy()
            printable_df = printable_df[[
                'product_id', 'production_line_id', 'product_name', 
                'production_line_name', 'month', 'year', 'real_hours_amount', 
                'operating_type_id', 'theoretical_hours_amount', 'delta_hours_amount'
            ]]

            # Prepare joined data dictionary
            joined_data = {
                'employee_df': employee_df,
                'employee_hours_df': employee_hours_df,
                'employee_groups_df': employee_groups_df,
                'employee_production_lines_df': employee_production_lines_df,
                'product_production_line_df': product_production_line_df,
                'demands_df': demands_df,
                'product_production_agg_df': product_production_agg_df,
                'printable_df': printable_df,
                'production_line_operating_type_df': production_line_operating_type_df
            }

            # Store joined data in data container
            if self.data_container:
                storage_key = self.data_container.store_stage_data(
                    stage_name="data_transformation",
                    data=joined_data,
                    metadata={
                        "process_id": self.process_id or "default_process",
                        "months": months,
                        "years": years,
                        "data_type": "joined_dataframes"
                    }
                )
                logger.info(f"Stored joined data in container with key: {storage_key}")
            
            # For backward compatibility and direct access,
            # update instance attributes
            for df_name, df in joined_data.items():
                setattr(self, df_name, df)
            
            self._data_transformed = True
            logger.info("Successfully joined dataframes")
            
            # Log shape information
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

    def assign_products(self, product_assignments: Dict[str, Dict[str, Any]]):
        """
        Method for assigning products to production lines.
        Inputs:
            - products_assignments: dictionary containing assignments by product_id, production_line_id, quantity
        """
        try:
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

            # Store the assignments in the data container
            if self.data_container:
                storage_key = self.data_container.store_stage_data(
                    stage_name="product_allocation",
                    data=df,
                    metadata={
                        "process_id": self.process_id or "default_process",
                        "assignment_count": len(df),
                        "data_type": "product_assignments"
                    }
                )
                logger.info(f"Stored product assignments in container with key: {storage_key}")
            
            # For backward compatibility and direct access
            self.product_production_line_assignments_df = df.copy()
            logger.info(f"Assignments: {self.product_production_line_assignments_df}")

            return True
        except Exception as e:
            logger.error(f"Error assigning products according to user decisions: {e}")
            return False
        
    def validate_product_assignments(self, product_assignments: Dict[str, Dict[str, Any]]):
        """
        Method for validating product assignments
        Inputs:
            - product_assignments: dictionary containing the product allocations from user decisions.
        """
        try:
            # Retrieve the product data from the data container or use instance attributes
            if self.data_container and self._data_transformed:
                try:
                    joined_data = self.data_container.retrieve_stage_data(
                        stage_name="data_transformation",
                        process_id=self.process_id or "default_process"
                    )
                    product_production_agg_df = joined_data.get('product_production_agg_df')
                    product_production_line_df = joined_data.get('product_production_line_df')
                except Exception as e:
                    logger.error(f"Error retrieving transformed data: {str(e)}")
                    # Fall back to instance attributes
                    product_production_agg_df = self.product_production_agg_df
                    product_production_line_df = self.product_production_line_df
            else:
                # Use instance attributes
                product_production_agg_df = self.product_production_agg_df
                product_production_line_df = self.product_production_line_df
            
            unexisting_products = []
            unexisting_prodlines = []
            unexisting_operating_types = []
            unexisting_combinations = []
            invalid_quantities = []
            
            # Convert unique values to appropriate types if needed
            unique_products = product_production_agg_df['product_id'].unique()
            unique_prodlines = product_production_agg_df['production_line_id'].unique()
            unique_operating_types = product_production_agg_df['operating_type_id'].unique()
            unique_product_prodline_comb = product_production_line_df[['product_id', 'production_line_id']].drop_duplicates()
            
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
                    unexisting_operating_types.append(operating_type_id)
                    
                if quantity <= 0:
                    invalid_quantities.append(quantity)

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
        
    def get_data_for_algorithm(self, algorithm_name: str):
        """
        Prepare data for a specific algorithm
        
        Args:
            algorithm_name: Name of the algorithm to prepare data for
            
        Returns:
            Dictionary with data prepared for the algorithm
        """
        try:
            # Retrieve the latest joined data if available
            if self.data_container and self._data_transformed:
                try:
                    joined_data = self.data_container.retrieve_stage_data(
                        stage_name="data_transformation",
                        process_id=self.process_id or "default_process"
                    )
                except Exception as e:
                    logger.error(f"Error retrieving joined data: {str(e)}")
                    joined_data = None
            else:
                joined_data = None
                
            # Prepare algorithm-specific data
            if algorithm_name.lower() == 'fillbags':
                # FillBags algorithm data
                if joined_data:
                    employee_df = joined_data.get('employee_df')
                    product_production_line_df = joined_data.get('product_production_line_df')
                    employee_production_lines_df = joined_data.get('employee_production_lines_df')
                else:
                    employee_df = self.employee_df
                    product_production_line_df = self.product_production_line_df
                    employee_production_lines_df = self.employee_production_lines_df
                
                return {
                    'emp_df': employee_df,
                    'prodl_df': product_production_line_df,
                    'emp_prodl_df': employee_production_lines_df
                }
                
            elif algorithm_name.lower() == 'lp':
                # LP algorithm data
                if joined_data:
                    employee_df = joined_data.get('employee_df')
                    production_line_df = self.production_lines_table  # Need the original for this one
                    employee_production_lines_df = joined_data.get('employee_production_lines_df')
                    product_production_agg_df = joined_data.get('product_production_agg_df')
                else:
                    employee_df = self.employee_df
                    production_line_df = self.production_lines_table
                    employee_production_lines_df = self.employee_production_lines_df
                    product_production_agg_df = self.product_production_agg_df
                
                return {
                    'employee_df': employee_df,
                    'production_line_df': production_line_df,
                    'employee_production_lines_df': employee_production_lines_df,
                    'product_production_agg_df': product_production_agg_df
                }
            
            else:
                logger.warning(f"Unknown algorithm: {algorithm_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing data for algorithm {algorithm_name}: {str(e)}")
            return None
            
    def store_algorithm_results(self, algorithm_name: str, results: Any):
        """
        Store algorithm results in the data container
        
        Args:
            algorithm_name: Name of the algorithm
            results: Algorithm results to store
            
        Returns:
            Storage key if successful, None otherwise
        """
        if not self.data_container:
            logger.warning("No data container available for storing algorithm results")
            return None
            
        try:
            storage_key = self.data_container.store_stage_data(
                stage_name="resource_allocation",
                data=results,
                metadata={
                    "process_id": self.process_id or "default_process",
                    "algorithm": algorithm_name,
                    "timestamp": datetime.now().isoformat(),
                    "data_type": "algorithm_results"
                }
            )
            logger.info(f"Stored {algorithm_name} results in container with key: {storage_key}")
            return storage_key
            
        except Exception as e:
            logger.error(f"Error storing algorithm results: {str(e)}")
            return None
            
    def retrieve_algorithm_results(self, algorithm_name: str):
        """
        Retrieve algorithm results from the data container
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm results if found, None otherwise
        """
        if not self.data_container:
            logger.warning("No data container available for retrieving algorithm results")
            return None
            
        try:
            # List available data
            available_data = self.data_container.list_available_data({
                "stage_name": "resource_allocation",
                "process_id": self.process_id or "default_process",
                "algorithm": algorithm_name
            })
            
            if not available_data:
                logger.warning(f"No results found for algorithm {algorithm_name}")
                return None
                
            # Sort by timestamp to get the most recent
            available_data.sort(key=lambda item: item['metadata']['timestamp'], reverse=True)
            latest_key = available_data[0]['key']
            
            # Retrieve the results
            results = self.data_container.retrieve_data(latest_key)
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving algorithm results: {str(e)}")
            return None
    
    def set_process_id(self, process_id: str):
        """Set the current process ID for data storage"""
        self.process_id = process_id
        logger.info(f"Set process ID to: {process_id}")