"""File containing the LpAlgo class"""

# Dependencies
import logging
import pandas as pd
import pulp
from base_data_project.algorithms.base import BaseAlgorithm
from base_data_project.data_manager.managers import BaseDataManager

# Local stuff
from src.config import PROJECT_NAME

# Get logger
logger = logging.getLogger(PROJECT_NAME)

class LpAlgo(BaseAlgorithm):
    """Class containing the base LP algorithm logic"""

    def __init__(self, data: BaseDataManager, parameters=None, algo_name=None):
        # Use the passed algo_name or default to "FillTheBags"
        actual_algo_name = algo_name or "LpAlgo"
        # Initialize the parent first
        super().__init__(algo_name=actual_algo_name, parameters=parameters)

        logger.info("Initializing LpAlgo object.")

        self.data = data
        self.status = ''
        self.temporal_space = 1
        self.availability = dict()
        self.auxiliary_values = dict()
        self.necessity = dict()
        self.results = None
        self.status = None

    def adapt_data(self):
        """
        Adapt data from core state to algorithm ready with enhanced logging
        """
        try:
            logger.info(f"Starting data adaptation to algorithm {self.algo_name}")
            
            # Log initial data dimensions
            logger.info(f"Initial employee_df: {self.data.employee_df.shape}")
            logger.info(f"Initial prodl_df: {self.data.production_line_df.shape}")
            logger.info(f"Initial employee_prodl_df: {self.data.emp_prodl_df.shape}")
            
            # Initialize important dataframes
            #employee_df = self.data.emp_df.copy()
            #prodl_df = self.data.prodl_df.copy()
            #employee_prodl_df = self.data.emp_prodl_df.copy()

            employee_df = self.data.employee_df.copy()
            production_line_df = self.data.production_line_df.copy()
            employee_production_lines_df = self.data.employee_production_lines_df.copy()
            product_production_agg_df = self.data.product_production_agg_df.copy()

            # Log DataFrame contents before processing
            logger.info(f"Production Line IDs: {production_line_df['ID'].tolist()}")
            logger.info(f"Unique Production Line IDs count: {production_line_df['ID'].nunique()}")
            
            # Get unique IDs
            employee_ids = sorted(employee_df['ID'].unique())
            prodline_ids = sorted(production_line_df['ID'].unique())
            
            logger.info(f"Unique Employee IDs: {len(employee_ids)}")
            logger.info(f"Unique Production Line IDs: {len(prodline_ids)}")
            
            # Create mappings to consecutive integers
            employee_map = {eid: i for i, eid in enumerate(employee_ids)}
            prodline_map = {pid: j for j, pid in enumerate(prodline_ids)}
            
            # Log the mappings
            logger.info(f"Employee ID to index mapping: {employee_map}")
            logger.info(f"Production Line ID to index mapping: {prodline_map}")
            
            # Create ranges
            i_values = range(len(employee_ids))
            j_values = range(len(prodline_ids))
            k_values = range(self.temporal_space)

            # Log range lengths
            logger.info(f"i_values range length: {len(i_values)}")
            logger.info(f"j_values range length: {len(j_values)}")
            logger.info(f"k_values range length: {len(k_values)}")
            
            # Create availability dictionary
            availability = {(i,j,k): 0 for i in i_values for j in j_values for k in k_values}
            
            # Create valid combinations using mapped values
            valid_pairs = set()
            for _, row in employee_production_lines_df.iterrows():
                emp_id = row['EMPLOYEE_ID']
                prod_id = row['PRODUCTION_LINE_ID']
                
                # Check if IDs exist in the maps
                if emp_id not in employee_map:
                    logger.warning(f"Employee ID {emp_id} not found in employee_map")
                    continue
                    
                if prod_id not in prodline_map:
                    logger.warning(f"Production Line ID {prod_id} not found in prodline_map")
                    continue
                    
                emp_idx = employee_map[emp_id]
                prod_idx = prodline_map[prod_id]
                valid_pairs.add((emp_idx, prod_idx))

            # Log valid pairs count
            logger.info(f"Valid employee-production line pairs: {len(valid_pairs)}")
            
            # Update availability
            availability = {
                (i,j,k): 1 if (i,j) in valid_pairs else availability[(i,j,k)]
                for i in i_values for j in j_values for k in k_values
            }

            # Create necessity dictionary
            necessity = {(j,k): 0 for j in j_values for k in k_values}
            necessity_dict_data = {}
            
            for _, row in production_line_df.iterrows():
                pid = row['ID']
                nec = row['NECESSITY']
                
                if pid not in prodline_map:
                    logger.warning(f"Production Line ID {pid} not found in prodline_map when creating necessity")
                    continue
                    
                j = prodline_map[pid]
                necessity_dict_data[j] = nec
            
            # Log necessity dictionary
            logger.info(f"Necessity dictionary entries: {len(necessity_dict_data)}")
            logger.info(f"Necessity values: {necessity_dict_data}")
            
            for k in k_values:
                for j in j_values:
                    if j in necessity_dict_data:
                        necessity[j,k] = necessity_dict_data[j]
                    else:
                        logger.warning(f"Missing necessity for production line index {j}")

            self.availability = availability
            self.necessity = necessity
            self.auxiliary_values = {
                'i_values': i_values,
                'j_values': j_values,
                'k_values': k_values,
                'employee_map': employee_map,
                'prodline_map': prodline_map
            }
            
            # Final data validation
            logger.info("Final data validation checks:")
            logger.info(f"Number of employees (i_values): {len(i_values)}")
            logger.info(f"Number of production lines (j_values): {len(j_values)}")
            logger.info(f"Number of time periods (k_values): {len(k_values)}")
            logger.info(f"Number of availability entries: {len(availability)}")
            logger.info(f"Number of necessity entries: {len(necessity)}")
            
            # Check for any missing values
            missing_necessity = [j for j in j_values if any(necessity.get((j,k), None) is None for k in k_values)]
            if missing_necessity:
                logger.warning(f"Missing necessity values for production line indices: {missing_necessity}")
            
            logger.info("Data transformation completed successfully")
        except Exception as e:
            logger.error(f"Error during data adaptation for {self.algo_name}: {str(e)}")
            raise
    
    def execute_algorithm(self):
        """
        Execute the linear programming algorithm
        """
        try:
            logger.info(f"Starting algorithm {self.algo_name} execution")

            # Get the value ranges from auxiliary values
            i_values = self.auxiliary_values['i_values']
            j_values = self.auxiliary_values['j_values']
            k_values = self.auxiliary_values['k_values']
            
            # Create the model
            prob = pulp.LpProblem("EmployeeScheduling", pulp.LpMinimize)
            
            # Decision variables
            x = pulp.LpVariable.dicts("schedule",
                                    ((i, j, k) for i in i_values 
                                            for j in j_values 
                                            for k in k_values),
                                    cat='Binary')
            
            # Slack variables
            under_staff = pulp.LpVariable.dicts("under",
                                            ((j, k) for j in j_values 
                                                    for k in k_values),
                                            lowBound=0)
            over_staff = pulp.LpVariable.dicts("over",
                                            ((j, k) for j in j_values 
                                                    for k in k_values),
                                            lowBound=0)
            
            # Objective function
            prob += (pulp.lpSum(under_staff[j,k] + over_staff[j,k] 
                    for j in j_values for k in k_values))
            
            # Constraints
            for i in i_values:
                for j in j_values:
                    for k in k_values:
                        if self.availability[(i,j,k)] == 0:
                            prob += x[i,j,k] == 0
            
            for i in i_values:
                for k in k_values:
                    prob += pulp.lpSum(x[i,j,k] for j in j_values) <= 1
            
            for j in j_values:
                for k in k_values:
                    prob += (pulp.lpSum(x[i,j,k] for i in i_values) + 
                            under_staff[j,k] - over_staff[j,k] == 
                            self.necessity[j,k])
            
            # Solve the problem
            logger.info("Solving LP problem...")
            prob.solve()
            
            # Store results
            self.status = pulp.LpStatus[prob.status]
            self.results = {
                'schedule': {(i,j,k): x[i,j,k].value() 
                            for i in i_values 
                            for j in j_values 
                            for k in k_values},
                'understaffing': {(j,k): under_staff[j,k].value() 
                                for j in j_values 
                                for k in k_values},
                'overstaffing': {(j,k): over_staff[j,k].value() 
                                for j in j_values 
                                for k in k_values},
                'objective_value': pulp.value(prob.objective)
            }

            # Store detailed results
            for j in j_values:
                for k in k_values:
                    self.assigned_employees = [
                        i for i in i_values if self.results['schedule'].get((i,j,k), 0) > 0.5
                    ]
                    # TODO: store these values or return them
            
            logger.info(f"LP algorithm completed successfully with status: {self.status}")
        except Exception as e:
            logger.error(f"Error during algorithm execution for {self.algo_name}: {str(e)}")
            raise
    
    def format_results(self, algorithm_results=None):
        """
        Convert results and store them
        """
        pass

    def run(self, common_data):
        """Algorithm Lp run method"""

        results = super().run(common_data)

        logger.info(f"Algorithm stage complete, data stored")

        return results