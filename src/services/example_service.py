#!/usr/bin/env python3
"""Example service implementation for the my_new_project project.

This service demonstrates how to use the process management framework to create
a coordinated multi-stage data processing flow.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Import base_data_project components
from base_data_project.data_manager.managers.base import BaseDataManager
from base_data_project.process_management.manager import ProcessManager
from base_data_project.algorithms.factory import AlgorithmFactory
from base_data_project.log_config import setup_logger

# Import project-specific components
from src.config import PROJECT_NAME, CONFIG

# Set up logger
logger = setup_logger(PROJECT_NAME, log_level=CONFIG.get('log_level', 'INFO'))

class ExampleService:
    """
    Example service class that demonstrates how to coordinate data management,
    process tracking, and algorithm execution.
    
    This service implements a complete process flow with multiple stages:
    1. Data Loading: Load data from sources
    2. Data Transformation: Clean and prepare the data
    3. Processing: Apply algorithms to the data
    4. Result Analysis: Analyze and save the results
    """

    def __init__(self, data_manager: BaseDataManager, process_manager: Optional[ProcessManager] = None):
        """
        Initialize the service with data and process managers.
        
        Args:
            data_manager: Data manager for data operations
            process_manager: Optional process manager for tracking
        """
        self.data_manager = data_manager
        self.process_manager = process_manager
        self.current_process_id = None
        
        # Initialize data placeholders
        self.raw_data = {}
        self.transformed_data = {}
        self.processing_results = {}
        
        logger.info("ExampleService initialized")

    def initialize_process(self, name: str, description: str) -> str:
        """
        Initialize a new process with the given name and description.
        
        Args:
            name: Process name
            description: Process description
            
        Returns:
            Process ID
        """
        logger.info(f"Initializing process: {name}")
        
        if self.process_manager:
            # Initialize the process with the process manager
            self.current_process_id = self.process_manager.initialize_process(name, description)
            logger.info(f"Process initialized with ID: {self.current_process_id}")
            return self.current_process_id
        else:
            # If no process manager, just log and use a placeholder ID
            logger.info("Process tracking disabled (no process manager)")
            self.current_process_id = f"no_tracking_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return self.current_process_id

    def execute_stage(self, stage: str, algorithm_name: Optional[str] = None, 
                     algorithm_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a specific stage in the process.
        
        Args:
            stage: Stage name to execute
            algorithm_name: Optional algorithm name for the processing stage
            algorithm_params: Optional parameters for the algorithm
            
        Returns:
            True if the stage executed successfully, False otherwise
        """
        try:
            logger.info(f"Executing process stage: {stage}")
            
            # Start stage in process manager if available
            if self.process_manager:
                self.process_manager.start_stage(stage, algorithm_name)
            
            # Execute the appropriate stage
            if stage == "data_loading":
                success = self._execute_data_loading_stage()
            elif stage == "data_transformation":
                success = self._execute_data_transformation_stage()
            elif stage == "processing":
                success = self._execute_processing_stage(algorithm_name, algorithm_params)
            elif stage == "result_analysis":
                success = self._execute_result_analysis_stage()
            else:
                logger.error(f"Unknown stage: {stage}")
                success = False
            
            # Complete stage in process manager if available
            if self.process_manager:
                result_data = {
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not success:
                    result_data["error"] = f"Stage {stage} failed"
                    
                self.process_manager.complete_stage(stage, success, result_data)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing stage {stage}: {str(e)}", exc_info=True)
            
            # Complete stage with failure in process manager if available
            if self.process_manager:
                self.process_manager.complete_stage(
                    stage, 
                    False, 
                    {"error": str(e), "timestamp": datetime.now().isoformat()}
                )
            
            return False

    def _execute_data_loading_stage(self) -> bool:
        """
        Execute the data loading stage.
        
        This stage loads data from the data source(s).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Executing data loading stage")
            
            # Get decisions from process manager if available
            load_entities = ["default_entity"]
            
            if self.process_manager:
                # Get selection decisions if available
                selections = self.process_manager.get_stage_decision(1, "selections")
                if selections and isinstance(selections, dict):
                    # Get entity selection from decisions
                    selected_entities = selections.get("selected_entities")
                    if selected_entities and isinstance(selected_entities, list):
                        load_entities = selected_entities
            
            # Track progress
            if self.process_manager:
                self.process_manager.track_progress(
                    "data_loading", 
                    0.1, 
                    "Starting data loading",
                    {"entities": load_entities}
                )
            
            # Load each entity
            self.raw_data = {}
            total_entities = len(load_entities)
            
            for i, entity in enumerate(load_entities):
                logger.info(f"Loading data for entity: {entity}")
                
                # Load data using the data manager
                try:
                    data = self.data_manager.load_data(entity)
                    self.raw_data[entity] = data
                    
                    # Log summary
                    if hasattr(data, 'shape'):
                        logger.info(f"Loaded {data.shape[0]} records for {entity}")
                    else:
                        logger.info(f"Loaded data for {entity}")
                    
                    # Track progress
                    if self.process_manager:
                        progress = (i + 1) / total_entities
                        self.process_manager.track_progress(
                            "data_loading", 
                            progress, 
                            f"Loaded {i+1}/{total_entities} entities",
                            {"current_entity": entity}
                        )
                        
                except Exception as e:
                    logger.error(f"Error loading data for entity {entity}: {str(e)}")
                    return False
            
            # Final progress update
            if self.process_manager:
                self.process_manager.track_progress(
                    "data_loading", 
                    1.0, 
                    "Data loading complete",
                    {"loaded_entities": list(self.raw_data.keys())}
                )
            
            logger.info("Data loading stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data loading stage: {str(e)}", exc_info=True)
            return False

    def _execute_data_transformation_stage(self) -> bool:
        """
        Execute the data transformation stage.
        
        This stage cleans and prepares the data for processing.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Executing data transformation stage")
            
            # Get transformation decisions from process manager if available
            transformations = {}
            
            if self.process_manager:
                # Get transformation decisions if available
                transformations = self.process_manager.get_stage_decision(2, "transformations") or {}
            
            # Track progress
            if self.process_manager:
                self.process_manager.track_progress(
                    "data_transformation", 
                    0.1, 
                    "Starting data transformation",
                    {"transformations": transformations}
                )
            
            # Apply transformations to each entity
            self.transformed_data = {}
            total_entities = len(self.raw_data)
            
            for i, (entity, data) in enumerate(self.raw_data.items()):
                logger.info(f"Transforming data for entity: {entity}")
                
                # Apply transformations
                # Note: This is a simplified example. In a real application,
                # you would implement specific transformations based on the entity type.
                try:
                    # Make a copy of the data to avoid modifying the original
                    transformed = data.copy() if hasattr(data, 'copy') else data
                    
                    # Example: Apply filtering if specified in transformations
                    if transformations.get("apply_filtering"):
                        # Example: Filter rows where a specific column meets a condition
                        filter_column = transformations.get("filter_column")
                        filter_value = transformations.get("filter_value")
                        
                        if hasattr(transformed, 'loc') and filter_column and filter_value is not None:
                            if filter_column in transformed.columns:
                                transformed = transformed.loc[transformed[filter_column] == filter_value]
                                logger.info(f"Filtered {entity} data on {filter_column}={filter_value}")
                    
                    # Store transformed data
                    self.transformed_data[entity] = transformed
                    
                    # Track progress
                    if self.process_manager:
                        progress = (i + 1) / total_entities
                        self.process_manager.track_progress(
                            "data_transformation", 
                            progress, 
                            f"Transformed {i+1}/{total_entities} entities",
                            {"current_entity": entity}
                        )
                        
                except Exception as e:
                    logger.error(f"Error transforming data for entity {entity}: {str(e)}")
                    return False
            
            # Final progress update
            if self.process_manager:
                self.process_manager.track_progress(
                    "data_transformation", 
                    1.0, 
                    "Data transformation complete",
                    {"transformed_entities": list(self.transformed_data.keys())}
                )
            
            logger.info("Data transformation stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data transformation stage: {str(e)}", exc_info=True)
            return False

    def _execute_processing_stage(self, algorithm_name: Optional[str] = None, 
                                algorithm_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute the processing stage using the specified algorithm.
        
        Args:
            algorithm_name: Name of the algorithm to use
            algorithm_params: Parameters for the algorithm
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If no algorithm specified, try to get from process manager
            if not algorithm_name and self.process_manager:
                # Get algorithm selection from decisions
                algorithm_selection = self.process_manager.get_stage_decision(3, "algorithm_selection") or {}
                algorithm_name = algorithm_selection.get("algorithm")
                
                # If parameters not provided, use from decisions
                if not algorithm_params:
                    algorithm_params = algorithm_selection.get("parameters", {})
            
            # Default to example_algorithm if still not specified
            if not algorithm_name:
                algorithm_name = "example_algorithm"
                
            logger.info(f"Executing processing stage with algorithm: {algorithm_name}")
            
            # Track progress
            if self.process_manager:
                self.process_manager.track_progress(
                    "processing", 
                    0.1, 
                    f"Starting processing with algorithm: {algorithm_name}",
                    {"algorithm": algorithm_name, "parameters": algorithm_params}
                )
            
            # Create the algorithm instance
            try:
                algorithm = AlgorithmFactory.create_algorithm(
                    algorithm_name=algorithm_name,
                    parameters=algorithm_params
                )
                
                # Track progress
                if self.process_manager:
                    self.process_manager.track_progress(
                        "processing", 
                        0.2, 
                        "Algorithm created, preparing data",
                        {"algorithm": algorithm_name}
                    )
                
                # Run the algorithm with the transformed data
                logger.info("Running algorithm")
                result = algorithm.run(self.transformed_data)
                
                # Track progress
                if self.process_manager:
                    self.process_manager.track_progress(
                        "processing", 
                        0.9, 
                        "Algorithm execution complete, finalizing",
                        {"algorithm": algorithm_name}
                    )
                
                # Store the results
                self.processing_results = result
                
                # Check result status
                if result.get("status") == "completed":
                    logger.info(f"Algorithm {algorithm_name} executed successfully")
                    
                    # Track metrics if available
                    metrics = result.get("metrics", {})
                    if metrics and self.process_manager:
                        self.process_manager.track_progress(
                            "processing", 
                            1.0, 
                            "Processing complete with metrics",
                            {"metrics": metrics}
                        )
                    
                    return True
                else:
                    error = result.get("error", "Unknown error")
                    logger.error(f"Algorithm execution failed: {error}")
                    return False
                
            except Exception as e:
                logger.error(f"Error creating or running algorithm {algorithm_name}: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error in processing stage: {str(e)}", exc_info=True)
            return False

    def _execute_result_analysis_stage(self) -> bool:
        """
        Execute the result analysis stage.
        
        This stage analyzes the processing results and saves the output.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Executing result analysis stage")
            
            # Get report generation decisions from process manager if available
            generate_report = True
            report_format = "csv"
            
            if self.process_manager:
                # Get report generation decisions if available
                report_generation = self.process_manager.get_stage_decision(4, "report_generation") or {}
                generate_report = report_generation.get("generate_report", True)
                report_format = report_generation.get("report_format", "csv")
            
            # Track progress
            if self.process_manager:
                self.process_manager.track_progress(
                    "result_analysis", 
                    0.1, 
                    "Starting result analysis",
                    {"generate_report": generate_report, "report_format": report_format}
                )
            
            # Extract the results
            if not self.processing_results:
                logger.error("No processing results to analyze")
                return False
            
            # Get the algorithm output data
            result_data = self.processing_results.get("data", {})
            metrics = self.processing_results.get("metrics", {})
            
            # Log the metrics
            if metrics:
                logger.info(f"Analysis metrics: {metrics}")
            
            # Generate report if requested
            if generate_report:
                logger.info(f"Generating report in {report_format} format")
                
                # Track progress
                if self.process_manager:
                    self.process_manager.track_progress(
                        "result_analysis", 
                        0.5, 
                        f"Generating report in {report_format} format",
                        {"format": report_format}
                    )
                
                # Example: Save the results
                try:
                    # Convert results to appropriate format for saving
                    import pandas as pd
                    
                    # Create a results dataframe (simplified example)
                    if isinstance(result_data, dict):
                        # Handle different types of result data
                        if any(isinstance(v, list) for v in result_data.values()):
                            # If there are lists in the values, use the first one
                            for key, value in result_data.items():
                                if isinstance(value, list):
                                    results_df = pd.DataFrame(value)
                                    break
                            else:
                                # If no lists found, create a simple dataframe
                                results_df = pd.DataFrame([result_data])
                        else:
                            # Simple dictionary
                            results_df = pd.DataFrame([result_data])
                    elif isinstance(result_data, list):
                        # List of results
                        results_df = pd.DataFrame(result_data)
                    else:
                        # Unknown format, create a simple metrics dataframe
                        results_df = pd.DataFrame([metrics])
                    
                    # Save the results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    entity_name = f"results_{timestamp}"
                    
                    self.data_manager.save_data(entity_name, results_df)
                    
                    logger.info(f"Results saved as {entity_name}")
                    
                    # Track progress
                    if self.process_manager:
                        self.process_manager.track_progress(
                            "result_analysis", 
                            0.9, 
                            "Report generated and saved",
                            {"entity_name": entity_name}
                        )
                    
                except Exception as e:
                    logger.error(f"Error saving results: {str(e)}")
                    return False
            
            # Final progress update
            if self.process_manager:
                self.process_manager.track_progress(
                    "result_analysis", 
                    1.0, 
                    "Result analysis complete",
                    {"metrics": metrics}
                )
            
            logger.info("Result analysis stage completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in result analysis stage: {str(e)}", exc_info=True)
            return False

    def finalize_process(self) -> None:
        """Finalize the process and clean up any resources."""
        logger.info("Finalizing process")
        
        # Nothing to do if no process manager
        if not self.process_manager:
            return
        
        # Log completion
        logger.info(f"Process {self.current_process_id} completed")

    def get_process_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current process.
        
        Returns:
            Dictionary with process summary information
        """
        if self.process_manager:
            return self.process_manager.get_process_summary()
        else:
            return {
                "status": "no_tracking",
                "process_id": self.current_process_id
            }

    def get_stage_decision(self, stage: int, decision_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific decision for a stage from the process manager.
        
        Args:
            stage: Stage number
            decision_name: Name of the decision
            
        Returns:
            Decision dictionary or None if not available
        """
        if self.process_manager:
            return self.process_manager.get_stage_decision(stage, decision_name)
        return None