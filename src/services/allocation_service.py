"""File containing the enhanced allocation service with improved process tracking"""

# Dependencies
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import logging
from datetime import datetime

# Local stuff
from src.config import CONFIG
from base_data_project.data_manager.managers import BaseDataManager
from base_data_project.process_management.manager import ProcessManager
from base_data_project.process_management.exceptions import ProcessManagementError
from src.helpers import parse_allocations
from src.algorithms.fill_bags import FillBagsAlgorithm
from src.algorithms.lp_algo import LpAlgo
from base_data_project.process_management.stage_handler import ProcessStageHandler
from src.models import AllocationData

logger = logging.getLogger('BagAllocationAlgo')

class AllocationService:
    """
    Service that orchestrates the bag allocation process with enhanced process tracking
    This service:
    1. Uses a data manager for data access
    2. Works with process manager for stage tracking 
    3. Implements domain-specific logic for bag allocation
    """

    def __init__(self, data_manager: BaseDataManager, process_manager: ProcessManager = None):
        """
        Initialize with required dependencies
        Args:
            data_manager: Data manager for accessing data sources
            process_manager: Optional process manager for tracking execution
        """
        self.data_manager = data_manager
        self.process_manager = process_manager

        # Initialize data structures - TODO: remove this
        self.emp_df = pd.DataFrame()
        self.prodl_df = pd.DataFrame()
        self.emp_prodl_df = pd.DataFrame()
        # TODO: Update to new dataframes
        self.bags = []
        self.balls = []

        # Process tracking 
        self.stage_handler = ProcessStageHandler(process_manager, CONFIG) if process_manager else None
        self.algorithm_results = {}

        logger.info("AllocationService initialized")

    def initialize_process(self, name: str, description: str = "") -> str:
        """
        Initialize a new allocation process
        
        Args:
            name: Name of the process
            description: Optional description of the process
            
        Returns:
            Process ID
        """
        logger.info(f"Initializing new process: {name}")
        
        if self.stage_handler:
            process_id = self.stage_handler.initialize_process(name, description)
            logger.info(f"Process initialized with ID: {process_id}")
            return process_id
        else:
            logger.warning("No process manager available, process will not be tracked")
            return "no_tracking"

    def execute_stage(self, stage_name: str, algorithm_name: str = None, algorithm_params: Dict[str, Any] = None) -> bool:
        """
        Execute a specific stage of the allocation process
        
        Args:
            stage_name: Name of the stage to execute
            algorithm_name: Optional algorithm name for stages with multiple algorithms
            algorithm_params: Optional parameters for the algorithm
            
        Returns:
            True if successful, False otherwise
        """
        if not self.stage_handler:
            logger.warning(f"Executing stage {stage_name} without process tracking")
            
            # Execute bare stage without tracking
            if stage_name == "data_loading":
                return self._execute_data_loading_bare()
            elif stage_name == "data_transformation":
                return self._execute_data_transformation_bare()
            elif stage_name == 'product_allocation':
                return self._execute_product_allocation_bare()
            elif stage_name == "resource_allocation":
                return self._execute_resource_allocation_bare(algorithm_name, algorithm_params or {})
            elif stage_name == "result_analysis":
                return self._execute_result_analysis_bare()
            else:
                logger.warning(f"No handler implemented for stage: {stage_name}")
                return False
        
        try:
            # Start stage execution with tracking
            stage = self.stage_handler.start_stage(stage_name, algorithm_name)
            
            # Record decision if algorithm parameters provided
            if algorithm_name and algorithm_params:
                self.stage_handler.record_stage_decision(stage_name, algorithm_name, algorithm_params)
            
            # Execute the appropriate stage handler
            if stage_name == "data_loading":
                success = self._execute_data_loading()
            elif stage_name == "data_transformation":
                success = self._execute_data_transformation()
            elif stage_name == "product_allocation":
                success = self._execute_product_allocation()
            elif stage_name == "resource_allocation":
                if not algorithm_name:
                    algorithms = stage.get("algorithms", ["fillbags"])
                    algorithm_name = algorithms[0]
                    
                success = self._execute_resource_allocation(algorithm_name, algorithm_params or {})
            elif stage_name == "result_analysis":
                success = self._execute_result_analysis()
            else:
                logger.warning(f"No handler implemented for stage: {stage_name}")
                success = False
            
            # Complete stage and record result
            result_data = {
                "success": success,
                "algorithm": algorithm_name if algorithm_name else None
            }
            
            if algorithm_name and algorithm_name in self.algorithm_results:
                result_algorithm = self.algorithm_results[algorithm_name]
                
                # Add algorithm-specific result data
                if hasattr(result_algorithm, 'filled'):
                    result_data["filled_status"] = result_algorithm.filled
                    
                if hasattr(result_algorithm, 'unused_balls_ids'):
                    result_data["unused_balls_count"] = len(result_algorithm.unused_balls_ids)
                    
                if hasattr(result_algorithm, 'status'):
                    result_data["solver_status"] = result_algorithm.status
            
            self.stage_handler.complete_stage(stage_name, success, result_data)
            return success
            
        except Exception as e:
            logger.error(f"Error executing stage {stage_name}: {str(e)}", exc_info=True)
            
            # Try to record failure if stage handler is available
            if self.stage_handler:
                try:
                    self.stage_handler.complete_stage(stage_name, False, {"error": str(e)})
                except:
                    pass
                    
            return False

    def _execute_data_loading_bare(self) -> bool:
        """Basic data loading without process tracking"""
        try:
            # Create a data container
            self.data = AllocationData()
            
            # Load data using the data manager
            success = self.data.load_from_data_manager(self.data_manager)
            if not success:
                return False
            
            # Validate the loaded data
            validation_result = self.data.validate()
            
            return validation_result
        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}", exc_info=True)
            return False

    def _execute_data_loading(self) -> bool:
        logger.info("Executing data loading stage")
        
        try:
            # Get stage decisions if available
            decisions = {}
            if self.stage_handler and self.process_manager:
                stage_sequence = self.stage_handler.stages['data_loading']['sequence']
                selections = self.process_manager.current_decisions.get(stage_sequence).get('selections', {})

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "data_loading", 
                    0.1, 
                    "Starting data loading process",
                    {"use_db": self.data_manager.config.get('db_url') is not None}
                )
            
            # Create a stage object for process tracking
            stage = type('ProcessStage', (), {
                'id': self.stage_handler.stages['data_loading']['id'], 
                'status': 'in_progress'
            })
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "data_loading", 
                    0.2, 
                    "Using existing data manager"
                )

            # Create a data container
            self.data = AllocationData()

            # Load data using the data manager
            success = self.data.load_from_data_manager(self.data_manager)
            if not success:
                return False            

            # Track progress
            self.stage_handler.track_progress(
                "data_loading", 
                0.3, 
                "Data loaded, filtering..."
            )

            # Get months and years from decisions
            months = selections.get('months', [1])
            years = selections.get('years', [2024])
            filter_result = self.data.filter_by_decision_time(months=months, years=years)
            if not filter_result:
                return False

            self.stage_handler.track_progress(
                "data_loading",
                0.4,
                "Data filtered, validating..."
            )

            # Validate the loaded data
            validation_result = self.data.validate()

            # TODO: add if not validated
            if not validation_result:
                pass

            self.stage_handler.track_progress(
                "data_loading",
                0.5,
                "Data validated"
            )

            # Final progress update
            self.stage_handler.track_progress(
                "data_loading", 
                1.0, 
                "Data loading complete",
                {
                    "validation_result": validation_result,
                    "filter_result": filter_result,
                    "data_shapes": {
                        "contract_types_table": self.data.contract_types_table.shape if self.data.contract_types_table is not None else None,
                        "demands_table": self.data.demands_table.shape if self.data.demands_table is not None else None,
                        "employees_table": self.data.employees_table.shape if self.data.employees_table is not None else None,
                        "employee_hours_table": self.data.employee_hours_table.shape if self.data.employee_hours_table is not None else None,
                        "employee_production_lines_table": self.data.employee_production_lines_table.shape if self.data.employee_production_lines_table is not None else None,
                        "employee_shift_assignments_table": self.data.employee_shift_assignments_table.shape if self.data.employee_shift_assignments_table is not None else None,
                        "groups_table": self.data.groups_table.shape if self.data.groups_table is not None else None,
                        "line_types_table": self.data.line_types_table.shape if self.data.line_types_table is not None else None,
                        "products_table": self.data.products_table.shape if self.data.products_table is not None else None,
                        "production_lines_table": self.data.production_lines_table.shape if self.data.production_lines_table is not None else None,
                        "production_lines_stats_table": self.data.production_lines_stats_table.shape if self.data.production_lines_stats_table is not None else None,
                        "product_production_line_assignments_table": self.data.product_production_line_assignments_table.shape if self.data.product_production_line_assignments_table is not None else None,
                        "product_production_line_table": self.data.product_production_line_table.shape if self.data.product_production_line_table is not None else None,
                        "sections_table": self.data.sections_table.shape if self.data.sections_table is not None else None,
                        "shifts_table": self.data.shifts_table.shape if self.data.shifts_table is not None else None,
                        "shift_types_table": self.data.shift_types_table.shape if self.data.shift_types_table is not None else None
                    }
                }
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "data_loading", 
                    0.0, 
                    f"Error loading data: {str(e)}"
                )
                
            return False

    def _execute_data_transformation_bare(self) -> bool:
        """Basic data transformation without process tracking"""
        # No specific implementation needed as transformation 
        # is handled within each algorithm
        return True

    def _execute_data_transformation(self) -> bool:
        """
        Execute the data transformation stage with process tracking
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing data transformation stage")
        
        try: 
            # Initialize variables
            decisions = {}
            filtering = {}
            excluded_employees_list = []
            excluded_lines_list = []
            adjustments_made_employees = 0
            adjustments_made_lines = 0
            stage_name = 'data_transformation'
            months = None
            years = None

            # Get stage decisions if available
            if self.stage_handler and self.process_manager:
                stage_sequence = self.stage_handler.stages[stage_name]['sequence']
                
                # Log all decisions for debugging
                logger.info(f"All process manager decisions: {self.process_manager.current_decisions}")
                logger.info(f"Looking for decisions at stage sequence: {stage_sequence}")
                
                if stage_sequence in self.process_manager.current_decisions:
                    decisions = self.process_manager.current_decisions[stage_sequence]
                    logger.info(f"Found decisions using direct access: {decisions}")
                    
                    # Get filtering information with proper nested access
                    if 'filtering' in decisions:
                        filtering = decisions['filtering']
                        logger.info(f"Found filtering decisions: {filtering}")

                # Get previous stage (data_loading) decisions to extract months and years
                data_loading_stage = self.stage_handler.stages['data_loading']
                data_loading_sequence = data_loading_stage['sequence']
                data_loading_decisions = self.process_manager.current_decisions.get(data_loading_sequence)

                if data_loading_decisions:
                    selections = data_loading_decisions.get('selections', {})
                    if selections:
                        months = selections.get('months')
                        years = selections.get('years')
                        logger.info(f"Retrieved months {months} and years {years} from data_loading decisions")

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.1,
                    message="Starting data transformation",
                    metadata={'decisions': decisions}
                )

            # Join dataframes from tables
            joining_result = self.data.join_dataframes(months, years)
            if not joining_result:
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.0,
                        message="Error joining dataframes. Returning false"
                    )
                return False

            # Extract excluded lists
            if filtering.get('apply_filtering', False):
                excluded_employees_list = filtering.get('filter_by_employees', [])
                excluded_lines_list = filtering.get('filter_by_lines', [])

                # Convert both lists to ints if they are not empty
                if excluded_employees_list != '':
                    excluded_employees_list = [int(x) for x in excluded_employees_list]
                if excluded_lines_list != '':
                    excluded_lines_list = [int(x) for x in excluded_lines_list]
                
                # Log what we found
                logger.info(f"Final filtering dictionary: {filtering}")
                logger.info(f"Apply filtering flag: {filtering.get('apply_filtering', False)}")
                logger.info(f"Excluded employees: {excluded_employees_list}")
                logger.info(f"Excluded lines: {excluded_lines_list}")

            # Apply filtering if enabled
            if filtering.get('apply_filtering', False) and self.data.employee_df is not None:
                # Track progress
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.2,
                        message="Applying employee/production line filtering",
                        metadata={"filtering": filtering}
                    )

                # Create working copies
                emp_df = self.data.employee_df.copy()
                emp_prodline_df = self.data.employee_production_lines_df.copy()
                production_line_df = self.data.production_lines_table.copy()
                prod_prodlines_df = self.data.product_production_line_df.copy()
                aggreg_data = self.data.product_production_agg_df.copy()
                printable_df = self.data.printable_df.copy() if hasattr(self.data, 'printable_df') else None

                # Employee filtering
                print(f"excluded_employees_list: {excluded_employees_list}")
                if excluded_employees_list and len(excluded_employees_list) > 0:
                    # Convert all IDs to string to ensure matching
                    excluded_employees_list = [int(id) for id in excluded_employees_list]
                    emp_df['id'] = emp_df['id'].astype(int)
                    
                    adjustments_made_employees = len(excluded_employees_list)
                    emp_df = emp_df[~emp_df['id'].isin(excluded_employees_list)]
                    self.data.employee_df = emp_df.copy()
                    logger.info(f"Made {adjustments_made_employees} changes to employee_df")

                    if self.stage_handler:
                        self.stage_handler.track_progress(
                            stage_name=stage_name,
                            progress=0.3,
                            message="Dataframe containing employee information filtered",
                            metadata={
                                'excluded_employees_list': excluded_employees_list
                            }
                        )

                # Production lines filtering
                if excluded_lines_list and len(excluded_lines_list) > 0:
                    # Convert all IDs to string to ensure matching
                    excluded_lines_list = [int(item) for item in excluded_lines_list]

                    emp_prodline_df['production_line_id'] = emp_prodline_df['production_line_id'].astype(int)
                    prod_prodlines_df['production_line_id'] = prod_prodlines_df['production_line_id'].astype(int)
                    production_line_df['id'] = production_line_df['id'].astype(int)
                    aggreg_data['production_line_id'] = aggreg_data['production_line_id'].astype(int)
                    
                    # Also filter the rintable_df
                    if printable_df is not None:
                        logger.info(f"Before filtering, printable_df has {len(printable_df)} rows")
                        printable_df['production_line_id'] = printable_df['production_line_id'].astype(int)
                        
                        # Log which production lines will be filtered out
                        logger.info(f"Excluding production lines: {excluded_lines_list}")
                        logger.info(f"Unique production line IDs in printable_df: {printable_df['production_line_id'].unique().tolist()}")
                        
                        # Filter the dataframe
                        printable_df = printable_df[~printable_df['production_line_id'].isin(excluded_lines_list)]
                        logger.info(f"After filtering, printable_df has {len(printable_df)} rows")
                        
                        # Update the object
                        self.data.printable_df = printable_df.copy()

                    adjustments_made_lines = len(excluded_lines_list)
                    production_line_df = production_line_df[~production_line_df['id'].isin(excluded_lines_list)]
                    emp_prodlines_df = emp_prodline_df[~emp_prodline_df['production_line_id'].isin(excluded_lines_list)]
                    prod_prodlines_df = prod_prodlines_df[~prod_prodlines_df['production_line_id'].isin(excluded_lines_list)]
                    aggreg_data = aggreg_data[~aggreg_data['production_line_id'].isin(excluded_lines_list)]
                    self.data.production_line_df = production_line_df.copy()
                    self.data.employee_production_lines_df = emp_prodlines_df.copy()
                    self.data.product_production_line_df = prod_prodlines_df.copy()
                    self.data.product_production_agg_df = aggreg_data.copy()
                    self.data.printable_df = self.data.product_production_agg_df.copy()
                    self.data.printable_df = self.data.printable_df[['product_id', 'production_line_id', 'product_name', 'production_line_name', 'month', 'year', 'real_hours_amount', 'operating_type_id', 'theoretical_hours_amount', 'delta_hours_amount']]

                    logger.info(f"Made {adjustments_made_lines} changes to production_line_df, employee_production_lines_df, product_production_line_df, product_production_agg_df")

                    self.data.printable_df.to_csv('C:/Users/antonio.alves/Documents/personal-stuff/projects/alocation-algo/operational-flexibility/data/output/printable_df.csv')

                    if self.stage_handler:
                        self.stage_handler.track_progress(
                            stage_name=stage_name,
                            progress=0.4,
                            message="Dataframe containing production line information filtered",
                            metadata={'excluded_lines_list': excluded_lines_list}
                        )

                # Final data transformation step: determine important values for algorithm running
                

                # Track progress
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name, 
                        progress=0.5, 
                        message=f"Employee adjustments applied to {adjustments_made_employees} employees. " + 
                                f"Line adjustments applied to {adjustments_made_lines} production lines."
                    )

            # Complete transformation progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=1.0,
                    message="Data transformation complete"
                )

            # Final progress update with metadata
            self.stage_handler.track_progress(
                stage_name=stage_name, 
                progress=1.0, 
                message="Data transformation complete",
                metadata={
                    "apply_filtering": filtering.get('apply_filtering', False),
                    "excluded_employees": excluded_employees_list,
                    "excluded_lines": excluded_lines_list,
                    "data_shapes": {
                        "employee_df": self.data.employee_df.shape if self.data.employee_df is not None else None,
                        "employee_hours_df": self.data.employee_hours_df.shape if self.data.employee_hours_df is not None else None,
                        "employee_groups_df": self.data.employee_groups_df.shape if self.data.employee_groups_df is not None else None,
                        "employee_production_lines_df": self.data.employee_production_lines_df.shape if self.data.employee_production_lines_df is not None else None,
                        "product_production_line_df": self.data.product_production_line_df.shape if self.data.product_production_line_df is not None else None,
                        "demands_df": self.data.demands_df.shape if self.data.demands_df is not None else None,
                        "product_production_agg_df": self.data.product_production_agg_df.shape if self.data.product_production_agg_df is not None else None
                    }
                }
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name="data_transformation", 
                    progress=0.0, 
                    message=f"Error in data transformation: {str(e)}"
                )
                
            return False

    def _execute_product_allocation_bare(self) -> bool:
        pass

    def _execute_product_allocation(self) -> bool:
        """
        Execute the product allocation stage with process tracking
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing product allocation stage")
        stage_name = 'product_allocation'
        try: 
            # Get stage decisions if available
            decisions = {}
            if self.stage_handler and self.process_manager:
                stage_sequence = self.stage_handler.stages[stage_name]['sequence']
                decisions = self.process_manager.current_decisions.get(stage_sequence, {})

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.1,
                    message="Starting product assignments validation.",
                    metadata={'decisions': decisions}
                )

            valid_assignments = self.data.validate_product_assignments(decisions.get('product_assignments', {}))
            if not valid_assignments:
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.0,
                        message="Invalid product assignments provided by user decisions."
                    )
                return False

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.2,
                    message="Starting valid product allocation",
                    metadata={'decisions': decisions}
                )

            assigning_result = self.data.assign_products(decisions.get('product_assignments', {}))

            if not assigning_result:
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.0,
                        message=f"Error assigning products according to user decisions",
                        metadata={'assigning_result': assigning_result}
                    )
                return False
            

            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.3,
                    message="Products allocation completed successfully",
                    metadata={'assigning_result': assigning_result}
                )
                    
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=1.0,
                    message="Product allocation process complete"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing product allocation: {e}")
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.0,
                    message=f"Error assigning products according tp user decisions",
                    metadata={
                        'assigning_result': assigning_result,
                        'data_shapes': {
                            'product_production_line_assignments_df': self.data.product_production_line_assignments_df.shape if self.data.product_production_line_assignments_df is not None else None
                        }
                    }
                )

            return False

    def _execute_resource_allocation_bare(self, algorithm_name: str, algorithm_params: Dict[str, Any]) -> bool:
        """Basic resource allocation without process tracking"""
        try:
            # Simple stage object for algorithm
            stage = type('ProcessStage', (), {
                'id': f'allocation_{algorithm_name}', 
                'status': 'in_progress'
            })
            
            # Execute the appropriate algorithm
            if algorithm_name == "fillbags":
                algorithm = FillBagsAlgorithm(algo_name=algorithm_name, data=self.data)
            elif algorithm_name == "lp":
                algorithm = LpAlgo(algo_name=algorithm_name, data=self.data)
            else:
                logger.error(f"Unknown algorithm: {algorithm_name}")
                return False
                
            # Run the algorithm with parameters
            algorithm.run(algorithm_params)
            
            # Store the results
            self.algorithm_results[algorithm_name] = algorithm
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing algorithm {algorithm_name}: {str(e)}", exc_info=True)
            return False

    def _execute_resource_allocation(self, algorithm_name: str, algorithm_params: Dict[str, Any]) -> bool:
        """
        Execute the resource allocation stage with a specific algorithm and process tracking
        
        Args:
            algorithm_name: Name of the algorithm to use
            algorithm_params: Parameters for the algorithm
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing resource allocation with algorithm: {algorithm_name}")
        
        try:
            # Track progress
            self.stage_handler.track_progress(
                "resource_allocation", 
                0.1, 
                f"Starting resource allocation with {algorithm_name}",
                {"parameters": algorithm_params}
            )
            
            # Create a stage object for process tracking
            stage = type('ProcessStage', (), {
                'id': self.stage_handler.stages['resource_allocation']['id'], 
                'status': 'in_progress'
            })
            
            # Track progress
            self.stage_handler.track_progress(
                "resource_allocation", 
                0.2, 
                f"Initializing {algorithm_name} algorithm"
            )
            
            # Execute the appropriate algorithm
            if algorithm_name == "fillbags":
                algorithm = FillBagsAlgorithm(algo_name=algorithm_name, data=self.data)
            elif algorithm_name == "lp":
                algorithm = LpAlgo(algo_name=algorithm_name, data=self.data)
            else:
                logger.error(f"Unknown algorithm: {algorithm_name}")
                
                # Track error
                self.stage_handler.track_progress(
                    "resource_allocation", 
                    0.0, 
                    f"Unknown algorithm: {algorithm_name}"
                )
                
                return False
                
            # Track progress
            self.stage_handler.track_progress(
                "resource_allocation", 
                0.3, 
                f"Running {algorithm_name} algorithm"
            )
                
            # Run the algorithm with parameters
            algorithm.run(algorithm_params)
            
            # Store the results
            self.algorithm_results[algorithm_name] = algorithm
            
            # Extract result summary data
            result_summary = {}
            
            if algorithm_name == "fillbags":
                result_summary = {
                    "filled_status": algorithm.filled,
                    "unused_balls_count": len(algorithm.unused_balls),
                    "unused_balls_capacity": algorithm.unused_balls_capacity,
                    "bag_count": len(algorithm.bag_allocations)
                }
            elif algorithm_name == "lp":
                result_summary = {
                    "solver_status": algorithm.status,
                    "objective_value": algorithm.results['objective_value'] if algorithm.results else None,
                }
            
            # Track completion
            self.stage_handler.track_progress(
                "resource_allocation", 
                1.0, 
                f"Resource allocation with {algorithm_name} complete",
                result_summary
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing algorithm {algorithm_name}: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "resource_allocation", 
                    0.0, 
                    f"Error executing {algorithm_name}: {str(e)}"
                )
                
            return False

    def _execute_result_analysis_bare(self) -> bool:
        """Basic result analysis without process tracking"""
        try:
            # For each algorithm that was run, save its output
            for algorithm_name, algorithm in self.algorithm_results.items():
                if hasattr(algorithm, 'save_output'):
                    algorithm.save_output()
                    logger.info(f"Saved output for algorithm: {algorithm_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in result analysis: {str(e)}", exc_info=True)
            return False

    def _execute_result_analysis(self) -> bool:
        """
        Execute the result analysis stage with process tracking and visualization
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing result analysis stage")
        # TODO: Add decision check for report generation output
        try:
            # Track progress
            self.stage_handler.track_progress(
                "result_analysis", 
                0.1, 
                "Starting result analysis"
            )
            
            # Track algorithms being analyzed
            self.stage_handler.track_progress(
                "result_analysis", 
                0.3, 
                "Analyzing algorithm results",
                {"algorithms": list(self.algorithm_results.keys())}
            )
            
            # For each algorithm that was run, save its output
            for algorithm_name, algorithm in self.algorithm_results.items():
                if hasattr(algorithm, 'save_output'):
                    # Track progress for each algorithm
                    self.stage_handler.track_progress(
                        "result_analysis", 
                        0.5, 
                        f"Processing output for {algorithm_name}"
                    )
                    
                    algorithm.save_output()
                    logger.info(f"Saved output for algorithm: {algorithm_name}")
            
            # Generate visualizations and HTML report if multiple algorithms were run
            if len(self.algorithm_results) > 0:
                # Track progress
                self.stage_handler.track_progress(
                    "result_analysis", 
                    0.7, 
                    "Generating visualizations and report"
                )
                
                # Import visualizer
                try:
                    from src.visualization.result_visualizer import ResultVisualizer
                    
                    # Create visualizer
                    visualizer = ResultVisualizer()
                    
                    # Generate report
                    process_summary = self.get_process_summary() if self.stage_handler else None
                    report_path = visualizer.generate_html_report(self.algorithm_results, process_summary)
                    
                    # Track report generation
                    if report_path and report_path != "Report generation failed":
                        self.stage_handler.track_progress(
                            "result_analysis", 
                            0.9, 
                            "Report generated successfully",
                            {"report_path": report_path}
                        )
                        logger.info(f"Generated HTML report: {report_path}")
                    else:
                        logger.warning("Failed to generate HTML report")
                except ImportError:
                    logger.warning("Visualization module not available, skipping report generation")
            
            # Track completion
            self.stage_handler.track_progress(
                "result_analysis", 
                1.0, 
                "Result analysis complete",
                {"algorithm_count": len(self.algorithm_results)}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in result analysis: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "result_analysis", 
                    0.0, 
                    f"Error in result analysis: {str(e)}"
                )
                
            return False

    def finalize_process(self) -> None:
        """
        Finalize the allocation process
        """
        logger.info("Finalizing allocation process")
        
        if self.stage_handler:
            # Get process summary for logging
            process_summary = self.stage_handler.get_process_summary()
            
            logger.info(f"Process finalized with "
                      f"{process_summary.get('status_counts', {}).get('completed', 0)} completed, "
                      f"{process_summary.get('status_counts', {}).get('failed', 0)} failed stages")
        
    def get_process_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current process
        
        Returns:
            Dictionary with process summary
        """
        if self.stage_handler:
            return self.stage_handler.get_process_summary()
        else:
            return {"status": "no_tracking"}
        
    def _validate_loaded_data(self, emp_df, prodl_df, emp_prodl_df) -> bool:
        """
        Basic validation of loaded data
        
        Returns:
            True if data is valid, False otherwise
        """
        # Check if all dataframes were loaded
        if emp_df is None or prodl_df is None or emp_prodl_df is None:
            logger.error("One or more required dataframes failed to load")
            return False
        
        # Check if dataframes have data
        if len(emp_df) == 0 or len(prodl_df) == 0 or len(emp_prodl_df) == 0:
            logger.error("One or more required dataframes is empty")
            return False
        
        # Check for required columns in each dataframe
        emp_required_cols = ['ID', 'CAPACITY_CONTRIBUTION']
        prodl_required_cols = ['ID', 'NECESSITY']
        emp_prodl_required_cols = ['EMPLOYEE_ID', 'PRODUCTION_LINE_ID']
        
        missing_cols = []
        for col in emp_required_cols:
            if col not in emp_df.columns:
                missing_cols.append(f"emp_df:{col}")
        
        for col in prodl_required_cols:
            if col not in prodl_df.columns:
                missing_cols.append(f"prodl_df:{col}")
                
        for col in emp_prodl_required_cols:
            if col not in emp_prodl_df.columns:
                missing_cols.append(f"emp_prodl_df:{col}")
        
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            return False
        
        logger.info("Data validation passed")
        return True