"""Allocation service with intermediate data storage integration"""

# Dependencies
from typing import Dict, Any, Optional, List, Tuple, Type
import pandas as pd
import logging
from datetime import datetime
import json

from base_data_project.process_management.stage_handler import ProcessStageHandler
from base_data_project.data_manager.managers import BaseDataManager
from base_data_project.process_management.exceptions import ProcessManagementError
from base_data_project.process_management.manager import ProcessManager
from base_data_project.service import BaseService
from base_data_project.storage.models import BaseDataModel
from base_data_project.storage.containers import BaseDataContainer, MemoryDataContainer
from base_data_project.storage.factory import DataContainerFactory

# Local stuff
from src.config import CONFIG, PROJECT_NAME
from src.helpers import parse_allocations
from src.algorithms.fill_bags import FillBagsAlgorithm
from src.algorithms.lp_algo import LpAlgo
from src.models import AllocationData

logger = logging.getLogger(PROJECT_NAME)

class AllocationService(BaseService):
    """
    Service that orchestrates the bag allocation process with enhanced process tracking
    and intermediate data storage.
    
    This service:
    1. Uses a data manager for data access
    2. Works with process manager for stage tracking 
    3. Implements domain-specific logic for bag allocation
    4. Stores intermediate results between stages
    """

    def __init__(self, data_manager: BaseDataManager, process_manager: Optional[ProcessManager] = None, 
                project_name: str = 'base_data_project', data_model_class: Optional[Type] = None):
        """
        Initialize with required dependencies
        Args:
            data_manager: Data manager for accessing data sources
            process_manager: Optional process manager for tracking execution
            project_name: Project name for logging
            data_model_class: Data model class to use (optional)
        """
        super().__init__(data_manager=data_manager, 
                        process_manager=process_manager, 
                        project_name=project_name, 
                        data_model_class=data_model_class)

        self.algorithm_results = {}
        
        # Ensure we have a data container
        if self.data_container is None:
            # Create a default memory container if none was provided
            storage_config = {
                'mode': 'memory',
                'project_name': project_name,
                'cleanup_policy': 'keep_latest'
            }
            self.data_container = DataContainerFactory.create_data_container(storage_config)
            logger.info(f"Created default memory data container for intermediate storage")
        
        # Initialize the data model if not done in parent
        if self.data_model is None:
            # If no data_model_class was provided, use the default AllocationData
            if data_model_class is None:
                from src.models import AllocationData
                data_model_class = AllocationData
                logger.info(f"Using default AllocationData model class")
            
            # Create an instance of the data model
            self.data_model = data_model_class(self.data_container)
            logger.info(f"Initialized data model: {self.data_model.__class__.__name__}")

        logger.info("AllocationService initialized with intermediate data storage")

    def _dispatch_stage(self, stage_name, algorithm_name = None, algorithm_params = None):
        """
        Dispatch execution to the appropriate stage handler based on stage name.
        
        Args:
            stage_name: Name of the stage to execute
            algorithm_name: Optional algorithm name for the resource allocation stage
            algorithm_params: Optional parameters for the algorithm
            
        Returns:
            True if the stage executed successfully, False otherwise
        """
        # Get stage from stage handler if available
        stage = None
        if self.stage_handler and stage_name in self.stage_handler.stages:
            stage = self.stage_handler.stages[stage_name]
            
        # Check if we have previously stored data for this stage
        stored_data = None
        if self.data_container:
            try:
                stored_data = self.data_container.retrieve_stage_data(stage_name, self.current_process_id)
                logger.info(f"Retrieved stored data for stage: {stage_name}")
            except KeyError:
                logger.info(f"No stored data found for stage: {stage_name}, will compute from scratch")
                stored_data = None
            except Exception as e:
                logger.warning(f"Error retrieving stored data for stage {stage_name}: {str(e)}")
                stored_data = None
                
        # Execute the appropriate stage handler
        if stage_name == "data_loading":
            return self._execute_data_loading(stored_data)
        elif stage_name == "data_transformation":
            return self._execute_data_transformation(stored_data)
        elif stage_name == "product_allocation":
            return self._execute_product_allocation(stored_data)
        elif stage_name == "resource_allocation":
            if not algorithm_name and stage:
                algorithms = stage.get("algorithms", ["fillbags"])
                algorithm_name = algorithms[0]
            return self._execute_resource_allocation(algorithm_name, algorithm_params or {}, stored_data)
        elif stage_name == "result_analysis":
            return self._execute_result_analysis(stored_data)
        else:
            logger.error(f"Unknown stage name: {stage_name}")
            return False

    def _execute_data_loading(self, stored_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute data loading stage with intermediate storage support
        
        Args:
            stored_data: Optional previously stored data for this stage
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing data loading stage")
        stage_name = 'data_loading'
        
        # If we have valid stored data, use it instead of recomputing
        if stored_data and isinstance(stored_data, dict) and 'status' in stored_data and stored_data['status'] == 'success':
            logger.info("Using stored data from previous data loading execution")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Using cached data loading results",
                    {"timestamp": stored_data.get('timestamp')}
                )
                
            # Restore data model state if applicable
            if hasattr(self.data_model, 'restore_state') and 'model_state' in stored_data:
                self.data_model.restore_state(stored_data['model_state'])
                logger.info("Restored data model state from storage")
                
            return True
        
        try:
            # Get stage decisions if available
            selections = {}
            if self.stage_handler and self.process_manager:
                stage_sequence = self.stage_handler.stages[stage_name]['sequence']
                
                # Get decisions from process manager
                decisions = self.process_manager.current_decisions.get(stage_sequence, {})
                selections = decisions.get('selections', {})

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.1, 
                    "Starting data loading process",
                    {"use_db": self.data_manager.config.get('db_url') is not None}
                )
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.2, 
                    "Using existing data manager"
                )

            # Load data using the data manager 
            success = self.data_model.load_from_data_manager(self.data_manager)
            if not success:
                return False            

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.3, 
                    "Data loaded, filtering..."
                )

            # Get months and years from decisions
            months = selections.get('months', [1])
            years = selections.get('years', [2024])
            filter_result = self.data_model.filter_by_decision_time(months=months, years=years)
            if not filter_result:
                return False

            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name,
                    0.4,
                    "Data filtered, validating..."
                )

            # Validate the loaded data
            validation_result = self.data_model.validate()
            if not validation_result:
                logger.warning("Data validation failed, but continuing execution")

            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name,
                    0.5,
                    "Data validated"
                )

            valid_joining_result = self.data_model.join_dataframes(months=months, years=years)
            if not valid_joining_result:
                return False

            # Store results in the intermediate storage
            if self.data_container:
                # Capture data model state if available
                model_state = None
                if hasattr(self.data_model, 'get_state'):
                    model_state = self.data_model.get_state()
                
                # Create result data
                result_data = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "filters": {
                        "months": months,
                        "years": years
                    },
                    "validation_result": validation_result,
                    "model_state": model_state
                }
                
                # Store in data container
                metadata = {
                    "process_id": self.current_process_id,
                    "stage_name": stage_name,
                    "decisions": {
                        "selections": selections
                    }
                }
                
                storage_key = self.data_container.store_stage_data(
                    stage_name=stage_name,
                    data=result_data,
                    metadata=metadata
                )
                logger.info(f"Stored data loading results with key: {storage_key}")

            # Final progress update
            if self.stage_handler:
                data_shapes = {}
                for attr_name in dir(self.data_model):
                    attr = getattr(self.data_model, attr_name)
                    if isinstance(attr, pd.DataFrame):
                        data_shapes[attr_name] = attr.shape
                
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Data loading complete",
                    {
                        "validation_result": validation_result,
                        "filter_result": filter_result,
                        "data_shapes": data_shapes
                    }
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.0, 
                    f"Error loading data: {str(e)}"
                )
            return False

    def _execute_data_transformation(self, stored_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute data transformation stage with intermediate storage support
        
        Args:
            stored_data: Optional previously stored data for this stage
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing data transformation stage")
        stage_name = 'data_transformation'
        
        # If we have valid stored data, use it instead of recomputing
        if stored_data and isinstance(stored_data, dict) and 'status' in stored_data and stored_data['status'] == 'success':
            logger.info("Using stored data from previous data transformation execution")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Using cached data transformation results",
                    {"timestamp": stored_data.get('timestamp')}
                )
                
            # Restore data model state if applicable
            if hasattr(self.data_model, 'restore_state') and 'model_state' in stored_data:
                self.data_model.restore_state(stored_data['model_state'])
                logger.info("Restored data model state from storage")
                
            return True
        
        try: 
            # Initialize variables
            decisions = {}
            filtering = {}
            excluded_employees_list = []
            excluded_lines_list = []
            adjustments_made_employees = 0
            adjustments_made_lines = 0
            months = None
            years = None

            # Get stage decisions if available
            if self.stage_handler and self.process_manager:
                stage = self.stage_handler.stages[stage_name]
                stage_sequence = stage['sequence']
                
                # Get decisions for current stage
                if stage_sequence in self.process_manager.current_decisions:
                    decisions = self.process_manager.current_decisions[stage_sequence]
                    logger.info(f"Found decisions for stage {stage_name}: {decisions}")
                    
                    # Get filtering information with proper nested access
                    if 'filtering' in decisions:
                        filtering = decisions['filtering']
                        logger.info(f"Found filtering decisions: {filtering}")

                # Get previous stage (data_loading) decisions to extract months and years
                data_loading_stage = self.stage_handler.stages['data_loading']
                data_loading_sequence = data_loading_stage['sequence']
                data_loading_decisions = self.process_manager.current_decisions.get(data_loading_sequence, {})

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

            # Verify we have joined data already
            if not hasattr(self.data_model, 'employee_df') or self.data_model.employee_df is None:
                # Try to get previously stored data from data_loading stage
                try:
                    data_loading_result = self.data_container.retrieve_stage_data('data_loading', self.current_process_id)
                    if (data_loading_result and isinstance(data_loading_result, dict) and 
                        'model_state' in data_loading_result and hasattr(self.data_model, 'restore_state')):
                        self.data_model.restore_state(data_loading_result['model_state'])
                        logger.info("Loaded data model state from data_loading stage")
                    else:
                        logger.warning("Could not restore data model from previous stage")
                        return False
                except Exception as e:
                    logger.error(f"Error loading previous stage data: {str(e)}")
                    return False

            # Join dataframes if needed
            if not hasattr(self.data_model, 'joining_complete') or not self.data_model.joining_complete:
                joining_result = self.data_model.join_dataframes(months, years)
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
                excluded_employees_list = filtering.get('excluded_employees', [])
                excluded_lines_list = filtering.get('excluded_lines', [])

                # Convert to integers if they are strings
                if isinstance(excluded_employees_list, str) and excluded_employees_list:
                    excluded_employees_list = [int(x) for x in excluded_employees_list.split(',')]
                elif isinstance(excluded_employees_list, list):
                    excluded_employees_list = [int(x) if isinstance(x, str) else x for x in excluded_employees_list]
                    
                if isinstance(excluded_lines_list, str) and excluded_lines_list:
                    excluded_lines_list = [int(x) for x in excluded_lines_list.split(',')]
                elif isinstance(excluded_lines_list, list):
                    excluded_lines_list = [int(x) if isinstance(x, str) else x for x in excluded_lines_list]
                
                # Log what we found
                logger.info(f"Final filtering dictionary: {filtering}")
                logger.info(f"Apply filtering flag: {filtering.get('apply_filtering', False)}")
                logger.info(f"Excluded employees: {excluded_employees_list}")
                logger.info(f"Excluded lines: {excluded_lines_list}")

            # Apply filtering if enabled
            if filtering.get('apply_filtering', False):
                # Track progress
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.2,
                        message="Applying employee/production line filtering",
                        metadata={"filtering": filtering}
                    )

                # Apply filtering through data model
                if hasattr(self.data_model, 'apply_filtering'):
                    filter_result = self.data_model.apply_filtering(
                        excluded_employees=excluded_employees_list,
                        excluded_lines=excluded_lines_list
                    )
                    
                    if not filter_result:
                        logger.warning("Filtering operation failed")
                        if self.stage_handler:
                            self.stage_handler.track_progress(
                                stage_name=stage_name,
                                progress=0.0,
                                message="Error applying filtering"
                            )
                        return False
                        
                    adjustments_made_employees = len(excluded_employees_list) if excluded_employees_list else 0
                    adjustments_made_lines = len(excluded_lines_list) if excluded_lines_list else 0
                    
                    # Track progress
                    if self.stage_handler:
                        self.stage_handler.track_progress(
                            stage_name=stage_name, 
                            progress=0.5, 
                            message=f"Employee adjustments applied to {adjustments_made_employees} employees. " + 
                                    f"Line adjustments applied to {adjustments_made_lines} production lines."
                        )
                else:
                    # Legacy filtering approach (keep this in case data_model doesn't support the method)
                    # [your existing filtering code would go here]
                    logger.warning("Data model doesn't support apply_filtering method")

            # Store results in the intermediate storage
            if self.data_container:
                # Capture data model state if available
                model_state = None
                if hasattr(self.data_model, 'get_state'):
                    model_state = self.data_model.get_state()
                
                # Create result data
                result_data = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "filtering": {
                        "apply_filtering": filtering.get('apply_filtering', False),
                        "excluded_employees": excluded_employees_list,
                        "excluded_lines": excluded_lines_list,
                    },
                    "adjustments": {
                        "employees": adjustments_made_employees,
                        "lines": adjustments_made_lines
                    },
                    "model_state": model_state
                }
                
                # Store in data container
                metadata = {
                    "process_id": self.current_process_id,
                    "stage_name": stage_name,
                    "decisions": decisions
                }
                
                storage_key = self.data_container.store_stage_data(
                    stage_name=stage_name,
                    data=result_data,
                    metadata=metadata
                )
                logger.info(f"Stored data transformation results with key: {storage_key}")

            # Complete transformation progress
            if self.stage_handler:
                # Get data shapes for tracking
                data_shapes = {}
                for attr_name in dir(self.data_model):
                    attr = getattr(self.data_model, attr_name)
                    if isinstance(attr, pd.DataFrame):
                        data_shapes[attr_name] = attr.shape
                
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=1.0,
                    message="Data transformation complete",
                    metadata={
                        "apply_filtering": filtering.get('apply_filtering', False),
                        "excluded_employees": excluded_employees_list,
                        "excluded_lines": excluded_lines_list,
                        "data_shapes": data_shapes
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

    def _execute_product_allocation(self, stored_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute product allocation stage with intermediate storage support
        
        Args:
            stored_data: Optional previously stored data for this stage
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing product allocation stage")
        stage_name = 'product_allocation'
        
        # If we have valid stored data, use it instead of recomputing
        if stored_data and isinstance(stored_data, dict) and 'status' in stored_data and stored_data['status'] == 'success':
            logger.info("Using stored data from previous product allocation execution")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Using cached product allocation results",
                    {"timestamp": stored_data.get('timestamp')}
                )
                
            # Restore data model state if applicable
            if hasattr(self.data_model, 'restore_state') and 'model_state' in stored_data:
                self.data_model.restore_state(stored_data['model_state'])
                logger.info("Restored data model state from storage")
                
            return True
            
        assigning_result = False
        try: 
            # Get stage decisions if available
            decisions = {}
            if self.stage_handler and self.process_manager:
                stage = self.stage_handler.stages[stage_name]
                stage_sequence = stage['sequence']
                
                # Get decisions from process manager
                decisions = self.process_manager.current_decisions.get(stage_sequence, {})

            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.1,
                    message="Starting product assignments validation.",
                    metadata={'decisions': decisions}
                )

            # Make sure we have the needed data from previous stages
            if not hasattr(self.data_model, 'validate_product_assignments'):
                # Try to load previous stage data
                try:
                    previous_stage = 'data_transformation'
                    previous_data = self.data_container.retrieve_stage_data(previous_stage, self.current_process_id)
                    if previous_data and 'model_state' in previous_data and hasattr(self.data_model, 'restore_state'):
                        self.data_model.restore_state(previous_data['model_state'])
                        logger.info(f"Loaded model state from {previous_stage} stage")
                    else:
                        logger.warning("Could not restore data model from previous stage")
                except Exception as e:
                    logger.error(f"Error loading previous stage data: {str(e)}")

            # Validate product assignments
            valid_assignments = self.data_model.validate_product_assignments(decisions.get('product_assignments', {}))
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

            # Assign products
            assigning_result = self.data_model.assign_products(decisions.get('product_assignments', {}))
            if not assigning_result:
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name=stage_name,
                        progress=0.0,
                        message=f"Error assigning products according to user decisions",
                        metadata={'assigning_result': assigning_result}
                    )
                return False
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name=stage_name,
                    progress=0.3,
                    message="Products allocation completed successfully",
                    metadata={'assigning_result': assigning_result}
                )
            
            # Store results in intermediate storage
            if self.data_container:
                # Capture data model state if available
                model_state = None
                if hasattr(self.data_model, 'get_state'):
                    model_state = self.data_model.get_state()
                
                # Create result data
                result_data = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "product_assignments": decisions.get('product_assignments', {}),
                    "assignment_result": assigning_result,
                    "model_state": model_state
                }
                
                # Store in data container
                metadata = {
                    "process_id": self.current_process_id,
                    "stage_name": stage_name,
                    "decisions": decisions
                }
                
                storage_key = self.data_container.store_stage_data(
                    stage_name=stage_name,
                    data=result_data,
                    metadata=metadata
                )
                logger.info(f"Stored product allocation results with key: {storage_key}")
                    
            # Final progress update
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
                    message=f"Error assigning products according to user decisions",
                    metadata={
                        'assigning_result': assigning_result,
                        'error': str(e)
                    }
                )

            return False

    def _execute_resource_allocation(self, algorithm_name: str, algorithm_params: Dict[str, Any], 
                                    stored_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute resource allocation stage with intermediate storage support
        
        Args:
            algorithm_name: Name of the algorithm to use
            algorithm_params: Parameters for the algorithm
            stored_data: Optional previously stored data for this stage
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing resource allocation with algorithm: {algorithm_name}")
        stage_name = 'resource_allocation'
        
        # If we have valid stored data for this exact algorithm, use it
        if (stored_data and isinstance(stored_data, dict) and 
            'status' in stored_data and stored_data['status'] == 'success' and
            'algorithm_name' in stored_data and stored_data['algorithm_name'] == algorithm_name):
            
            logger.info(f"Using stored data from previous {algorithm_name} execution")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    f"Using cached {algorithm_name} results",
                    {"timestamp": stored_data.get('timestamp')}
                )
                
            # Restore algorithm results if available
            if 'algorithm_results' in stored_data:
                self.algorithm_results[algorithm_name] = stored_data['algorithm_results']
                logger.info(f"Restored {algorithm_name} results from storage")
                
            # Restore data model state if applicable
            if hasattr(self.data_model, 'restore_state') and 'model_state' in stored_data:
                self.data_model.restore_state(stored_data['model_state'])
                logger.info("Restored data model state from storage")
                
            return True
        
        try:
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.1, 
                    f"Starting resource allocation with {algorithm_name}",
                    {"parameters": algorithm_params}
                )
            
            # Make sure we have data from previous stages
            if not hasattr(self.data_model, 'employee_df') or self.data_model.employee_df is None:
                # Try to load previous stage data
                try:
                    previous_stage = 'product_allocation'
                    previous_data = self.data_container.retrieve_stage_data(previous_stage, self.current_process_id)
                    if previous_data and 'model_state' in previous_data and hasattr(self.data_model, 'restore_state'):
                        self.data_model.restore_state(previous_data['model_state'])
                        logger.info(f"Loaded model state from {previous_stage} stage")
                    else:
                        logger.warning("Could not restore data model from previous stage")
                        
                        # Try loading from data_transformation stage
                        previous_stage = 'data_transformation'
                        previous_data = self.data_container.retrieve_stage_data(previous_stage, self.current_process_id)
                        if previous_data and 'model_state' in previous_data and hasattr(self.data_model, 'restore_state'):
                            self.data_model.restore_state(previous_data['model_state'])
                            logger.info(f"Loaded model state from {previous_stage} stage")
                        else:
                            logger.warning("Could not restore data model from previous stages")
                            return False
                except Exception as e:
                    logger.error(f"Error loading previous stage data: {str(e)}")
                    return False
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.2, 
                    f"Initializing {algorithm_name} algorithm"
                )
            
            # Execute the appropriate algorithm
            if algorithm_name == "fillbags":
                algorithm = FillBagsAlgorithm(algo_name=algorithm_name, data=self.data_model)
            elif algorithm_name == "lp":
                algorithm = LpAlgo(algo_name=algorithm_name, data=self.data_model)
            else:
                logger.error(f"Unknown algorithm: {algorithm_name}")

                # Track error
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name, 
                        0.0, 
                        f"Unknown algorithm: {algorithm_name}"
                    )    
                return False
                
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
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
                    "unused_balls_count": len(algorithm.unused_balls_ids) if hasattr(algorithm, 'unused_balls_ids') else 0,
                    "unused_balls_capacity": algorithm.unused_balls_capacity,
                    "bag_count": len(algorithm.bag_allocations)
                }
            elif algorithm_name == "lp":
                result_summary = {
                    "solver_status": algorithm.status,
                    "objective_value": algorithm.results['objective_value'] if algorithm.results else None,
                }
            
            # Store results in intermediate storage
            if self.data_container:
                # Capture data model state if available
                model_state = None
                if hasattr(self.data_model, 'get_state'):
                    model_state = self.data_model.get_state()
                
                # Serialize algorithm results for storage
                # We can't store the whole algorithm object, so extract key data
                algorithm_data = {
                    "status": algorithm.status,
                    "execution_time": algorithm.execution_time,
                    "result_summary": result_summary
                }
                
                # Add algorithm-specific data
                if algorithm_name == "fillbags":
                    algorithm_data.update({
                        "filled": algorithm.filled,
                        "unused_balls_ids": algorithm.unused_balls_ids if hasattr(algorithm, 'unused_balls_ids') else [],
                        "unused_balls_capacity": algorithm.unused_balls_capacity,
                        "bag_allocations": algorithm.bag_allocations
                    })
                elif algorithm_name == "lp":
                    if hasattr(algorithm, 'results') and algorithm.results:
                        algorithm_data["results"] = {
                            "objective_value": algorithm.results.get('objective_value'),
                            "status": algorithm.results.get('status')
                            # Add other LP-specific results that are needed
                        }
                
                # Create result data
                result_data = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "algorithm_name": algorithm_name,
                    "algorithm_params": algorithm_params,
                    "algorithm_results": algorithm_data,
                    "result_summary": result_summary,
                    "model_state": model_state
                }
                
                # Store in data container
                metadata = {
                    "process_id": self.current_process_id,
                    "stage_name": stage_name,
                    "algorithm": algorithm_name
                }
                
                storage_key = self.data_container.store_stage_data(
                    stage_name=f"{stage_name}_{algorithm_name}",
                    data=result_data,
                    metadata=metadata
                )
                logger.info(f"Stored {algorithm_name} results with key: {storage_key}")
            
            # Track completion
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
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
                    stage_name, 
                    0.0, 
                    f"Error executing {algorithm_name}: {str(e)}"
                )
                
            return False

    def _execute_result_analysis(self, stored_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute result analysis stage with intermediate storage support
        
        Args:
            stored_data: Optional previously stored data for this stage
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Executing result analysis stage")
        stage_name = 'result_analysis'
        
        # If we have valid stored data, use it instead of recomputing
        if stored_data and isinstance(stored_data, dict) and 'status' in stored_data and stored_data['status'] == 'success':
            logger.info("Using stored data from previous result analysis execution")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Using cached result analysis results",
                    {"timestamp": stored_data.get('timestamp')}
                )
                
            # Restore report path if available
            if 'report_path' in stored_data:
                report_path = stored_data['report_path']
                logger.info(f"Restored report path from storage: {report_path}")
                
            return True
            
        try:
            # Get decisions for stage if available
            decisions = {}
            generate_report = True  # Default to true
            
            if self.stage_handler and self.process_manager:
                stage = self.stage_handler.stages[stage_name]
                stage_sequence = stage['sequence']
                
                # Get decisions from process manager
                decisions = self.process_manager.current_decisions.get(stage_sequence, {})
                
                # Check if report generation is disabled in decisions
                if 'changes' in decisions and 'generate_report' in decisions['changes']:
                    generate_report = decisions['changes']['generate_report']
            
            # Check if we have algorithm results
            if not self.algorithm_results:
                # Try to load algorithm results from storage
                algorithm_names = ["fillbags", "lp"]  # Add all potential algorithms
                
                for algorithm_name in algorithm_names:
                    try:
                        algo_stage_name = f"resource_allocation_{algorithm_name}"
                        algo_data = self.data_container.retrieve_stage_data(algo_stage_name, self.current_process_id)
                        
                        if algo_data and 'algorithm_results' in algo_data:
                            # Create a mock algorithm object with results
                            class MockAlgorithm:
                                def __init__(self, data):
                                    for key, value in data.items():
                                        setattr(self, key, value)
                            
                            # Create mock algorithm with stored data
                            mock_algo = MockAlgorithm(algo_data['algorithm_results'])
                            self.algorithm_results[algorithm_name] = mock_algo
                            
                            logger.info(f"Restored {algorithm_name} results from storage")
                    except (KeyError, Exception) as e:
                        logger.info(f"No stored results found for {algorithm_name}: {str(e)}")
            
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.1, 
                    "Starting result analysis"
                )
            
            # Track algorithms being analyzed
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.3, 
                    "Analyzing algorithm results",
                    {"algorithms": list(self.algorithm_results.keys())}
                )
            
            # For each algorithm that was run, save its output
            for algorithm_name, algorithm in self.algorithm_results.items():
                if hasattr(algorithm, 'save_output'):
                    # Track progress for each algorithm
                    if self.stage_handler:
                        self.stage_handler.track_progress(
                            stage_name, 
                            0.5, 
                            f"Processing output for {algorithm_name}"
                        )
                    
                    algorithm.save_output()
                    logger.info(f"Saved output for algorithm: {algorithm_name}")
            
            # Generate visualizations and HTML report if requested
            report_path = None
            if generate_report and len(self.algorithm_results) > 0:
                # Track progress
                if self.stage_handler:
                    self.stage_handler.track_progress(
                        stage_name, 
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
                        if self.stage_handler:
                            self.stage_handler.track_progress(
                                stage_name, 
                                0.9, 
                                "Report generated successfully",
                                {"report_path": report_path}
                            )
                        logger.info(f"Generated HTML report: {report_path}")
                    else:
                        logger.warning("Failed to generate HTML report")
                except ImportError:
                    logger.warning("Visualization module not available, skipping report generation")
            
            # Store results in intermediate storage
            if self.data_container:
                # Create result data
                result_data = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "algorithm_count": len(self.algorithm_results),
                    "algorithms_analyzed": list(self.algorithm_results.keys()),
                    "generate_report": generate_report,
                    "report_path": report_path
                }
                
                # Store in data container
                metadata = {
                    "process_id": self.current_process_id,
                    "stage_name": stage_name,
                    "decisions": decisions
                }
                
                storage_key = self.data_container.store_stage_data(
                    stage_name=stage_name,
                    data=result_data,
                    metadata=metadata
                )
                logger.info(f"Stored result analysis with key: {storage_key}")
            
            # Track completion
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    1.0, 
                    "Result analysis complete",
                    {"algorithm_count": len(self.algorithm_results),
                     "report_path": report_path if report_path else None}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in result analysis: {str(e)}", exc_info=True)
            
            # Track error
            if self.stage_handler:
                self.stage_handler.track_progress(
                    stage_name, 
                    0.0, 
                    f"Error in result analysis: {str(e)}"
                )
                
            return False
    
    def get_decisions_for_stage(self, stage_name: str) -> Tuple[int, Dict[str, Any]]:
        """
        Get decisions for a specific stage by stage name.
        
        Args:
            stage_name: Name of the stage to get decisions for
            
        Returns:
            Tuple of (stage_sequence, decisions_dict)
        """
        stage_sequence = None
        decisions = {}
        
        if not self.stage_handler or not self.process_manager:
            return 0, {}
            
        stage = self.stage_handler.stages.get(stage_name)
        if not stage:
            logger.warning(f"Stage '{stage_name}' not found in stage handler")
            return 0, {}
            
        stage_sequence = stage.get('sequence')
        if stage_sequence is None:
            logger.warning(f"No sequence found for stage '{stage_name}'")
            return 0, {}
            
        decisions = self.process_manager.current_decisions.get(stage_sequence, {})
        return stage_sequence, decisions
    
    def list_available_data(self, stage_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available intermediate data.
        
        Args:
            stage_name: Optional stage name to filter by
            
        Returns:
            List of available data summaries
        """
        if not self.data_container:
            return []
            
        filters = {}
        if self.current_process_id:
            filters['process_id'] = self.current_process_id
            
        if stage_name:
            filters['stage_name'] = stage_name
            
        try:
            return self.data_container.list_available_data(filters)
        except Exception as e:
            logger.error(f"Error listing available data: {str(e)}")
            return []
    
    def cleanup_stored_data(self, policy: Optional[str] = None) -> bool:
        """
        Clean up stored intermediate data based on policy.
        
        Args:
            policy: Cleanup policy ('keep_none', 'keep_latest', 'keep_all')
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        if not self.data_container:
            return False
            
        try:
            self.data_container.cleanup(policy)
            logger.info(f"Cleaned up stored data with policy: {policy}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up stored data: {str(e)}")
            return False