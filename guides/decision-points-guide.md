# Adding Decision Points to Resource Allocation System - Implementation Guide

This guide provides a comprehensive approach to implementing decision points at each stage of your Resource Allocation System's interactive workflow.

## Table of Contents

1. [Understanding Decision Points](#understanding-decision-points)
2. [Architecture Overview](#architecture-overview)
3. [Decision Point Implementation by Stage](#decision-point-implementation-by-stage)
   - [Stage 1: Data Loading Decisions](#stage-1-data-loading-decisions)
   - [Stage 2: Data Transformation Decisions](#stage-2-data-transformation-decisions)
   - [Stage 3: Resource Allocation Decisions](#stage-3-resource-allocation-decisions)
   - [Stage 4: Result Analysis Decisions](#stage-4-result-analysis-decisions)
4. [Modifying the Process Stage Handler](#modifying-the-process-stage-handler)
5. [Updating the Interactive Process](#updating-the-interactive-process)
6. [Testing Your Implementation](#testing-your-implementation)
7. [Extending the System](#extending-the-system)

## Understanding Decision Points

Decision points are moments in your process where the user can provide input to guide the execution. Each decision:

- Has a specific purpose (e.g., choosing an algorithm)
- May have default values
- Should be tracked for reproducibility
- Might affect subsequent decisions

## Architecture Overview

Your system already has a solid foundation for implementing decision points:

- **ProcessManager**: Tracks decisions and their impacts
- **ProcessStageHandler**: Manages stage progression and captures decisions
- **AllocationService**: Orchestrates the overall process

The interactive CLI provides the UI for capturing user decisions.

## Decision Point Implementation by Stage

### Stage 1: Data Loading Decisions

Data loading decisions control how data is imported into the system.

```python
# Add to run_allocation_process_interactive in main.py
if current_stage == "data_loading":
    click.echo(click.style("\nData Loading Options:", fg="cyan"))
    
    # Choose data validation level
    validation_level = click.prompt(
        "Choose validation level",
        type=click.Choice(['basic', 'strict', 'permissive']),
        default='basic'
    )
    
    # Whether to perform sample data checks
    sample_checks = click.confirm(
        "Perform sample data checks?", 
        default=True
    )
    
    data_params = {
        'validation_level': validation_level,
        'perform_sample_checks': sample_checks
    }
    
    # Pass the params to the stage execution
    success = allocation_service.execute_stage(
        current_stage, 
        algorithm_params=data_params
    )
```

#### Implementation in AllocationService

Add a new method to handle data loading decisions:

```python
def _execute_data_loading(self) -> bool:
    """Execute data loading with decision parameters"""
    try:
        # Get decision parameters if available
        params = self.stage_handler.stages.get("data_loading", {}).get("decisions", {})
        validation_level = params.get("validation_level", "basic")
        perform_sample_checks = params.get("perform_sample_checks", True)
        
        # Log decisions
        logger.info(f"Executing data loading with validation level: {validation_level}")
        logger.info(f"Sample checks enabled: {perform_sample_checks}")
        
        # Create data container with parameters
        self.data = AllocationData(
            validation_level=validation_level,
            perform_sample_checks=perform_sample_checks
        )
        
        # Continue with existing implementation...
        # ...
    except Exception as e:
        logger.error(f"Error in data loading: {str(e)}")
        return False
```

### Stage 2: Data Transformation Decisions

Data transformation decisions determine how data is processed before allocation.

```python
# Add to run_allocation_process_interactive in main.py
if current_stage == "data_transformation":
    click.echo(click.style("\nData Transformation Options:", fg="cyan"))
    
    # Choose filtering criteria
    include_inactive = click.confirm(
        "Include inactive employees?",
        default=False
    )
    
    min_capacity = click.prompt(
        "Minimum capacity contribution",
        type=float,
        default=0.1
    )
    
    normalization = click.prompt(
        "Choose capacity normalization method",
        type=click.Choice(['none', 'min-max', 'z-score']),
        default='none'
    )
    
    transform_params = {
        'include_inactive': include_inactive,
        'min_capacity': min_capacity,
        'normalization': normalization
    }
    
    # Pass the params to the stage execution
    success = allocation_service.execute_stage(
        current_stage,
        algorithm_params=transform_params
    )
```

#### Implementation in AllocationService

```python
def _execute_data_transformation(self) -> bool:
    """Execute data transformation with decision parameters"""
    try:
        # Get decision parameters if available
        params = self.stage_handler.stages.get("data_transformation", {}).get("decisions", {})
        include_inactive = params.get("include_inactive", False)
        min_capacity = params.get("min_capacity", 0.1)
        normalization = params.get("normalization", "none")
        
        # Log decisions
        logger.info(f"Data transformation parameters: include_inactive={include_inactive}, " +
                    f"min_capacity={min_capacity}, normalization={normalization}")
        
        # Apply filtering if needed
        if not include_inactive:
            # Filter out inactive employees (example)
            if hasattr(self.data.emp_df, 'ACTIVE'):
                self.data.emp_df = self.data.emp_df[self.data.emp_df['ACTIVE'] == True]
        
        # Apply minimum capacity filter
        if min_capacity > 0:
            self.data.emp_df = self.data.emp_df[
                self.data.emp_df['CAPACITY_CONTRIBUTION'] >= min_capacity
            ]
        
        # Apply normalization if requested
        if normalization != 'none':
            if normalization == 'min-max':
                # Min-max normalization to [0,1] range
                min_val = self.data.emp_df['CAPACITY_CONTRIBUTION'].min()
                max_val = self.data.emp_df['CAPACITY_CONTRIBUTION'].max()
                range_val = max_val - min_val
                if range_val > 0:
                    self.data.emp_df['CAPACITY_CONTRIBUTION'] = (
                        (self.data.emp_df['CAPACITY_CONTRIBUTION'] - min_val) / range_val
                    )
            elif normalization == 'z-score':
                # Z-score normalization
                mean = self.data.emp_df['CAPACITY_CONTRIBUTION'].mean()
                std = self.data.emp_df['CAPACITY_CONTRIBUTION'].std()
                if std > 0:
                    self.data.emp_df['CAPACITY_CONTRIBUTION'] = (
                        (self.data.emp_df['CAPACITY_CONTRIBUTION'] - mean) / std
                    )
        
        # Track progress and completion
        if self.stage_handler:
            self.stage_handler.track_progress(
                "data_transformation", 
                1.0, 
                "Data transformation complete",
                {
                    "rows_remaining": len(self.data.emp_df),
                    "normalization": normalization
                }
            )
        
        return True
    
    except Exception as e:
        logger.error(f"Error in data transformation: {str(e)}")
        if self.stage_handler:
            self.stage_handler.track_progress(
                "data_transformation", 
                0.0, 
                f"Error in data transformation: {str(e)}"
            )
        return False
```

### Stage 3: Resource Allocation Decisions

Your system already has resource allocation decisions well implemented, with algorithm selection and algorithm-specific parameters. Here's how to extend it further:

```python
# Add to the existing resource_allocation section in run_allocation_process_interactive
# This extends your existing implementation with more options

# Add advanced parameters for both algorithms
advanced_options = click.confirm("Configure advanced parameters?", default=False)

if advanced_options:
    if algorithm_name == "fillbags":
        # Add additional FillBags parameters
        max_iterations = click.prompt(
            "Maximum iterations",
            type=int,
            default=1000
        )
        
        convergence_threshold = click.prompt(
            "Convergence threshold",
            type=float,
            default=0.001
        )
        
        algorithm_params.update({
            'max_iterations': max_iterations,
            'convergence_threshold': convergence_threshold
        })
    
    elif algorithm_name == "lp":
        # Add additional LP parameters
        solver_type = click.prompt(
            "Solver type",
            type=click.Choice(['default', 'PULP_CBC_CMD', 'GLPK']),
            default='default'
        )
        
        time_limit = click.prompt(
            "Solver time limit (seconds)",
            type=int,
            default=60
        )
        
        algorithm_params.update({
            'solver_type': solver_type,
            'time_limit': time_limit
        })
```

#### Update FillBagsAlgorithm to use advanced parameters

```python
def execute_algorithm(self, adapted_data=None):
    """Execute the bag filling algorithm with advanced parameters"""
    try:
        # Get advanced parameters if provided
        max_iterations = self.parameters.get('max_iterations', 1000)
        convergence_threshold = self.parameters.get('convergence_threshold', 0.001)
        
        logger.info(f"Running with max_iterations={max_iterations}, threshold={convergence_threshold}")
        
        # Continue with existing implementation, adding iteration checks...
        # ...
    except Exception as e:
        logger.error(f"Error during bag filling: {str(e)}")
        raise
```

#### Update LpAlgo to use advanced parameters

```python
def execute_algorithm(self, adapted_data=None):
    """Execute the linear programming algorithm with advanced parameters"""
    try:
        # Get advanced parameters if provided
        solver_type = self.parameters.get('solver_type', 'default')
        time_limit = self.parameters.get('time_limit', 60)
        
        logger.info(f"Running LP with solver={solver_type}, time_limit={time_limit}")
        
        # Create the model
        prob = pulp.LpProblem("EmployeeScheduling", pulp.LpMinimize)
        
        # Configure solver based on parameters
        if solver_type != 'default':
            if solver_type == 'PULP_CBC_CMD':
                solver = pulp.PULP_CBC_CMD(timeLimit=time_limit)
            elif solver_type == 'GLPK':
                solver = pulp.GLPK(timeLimit=time_limit)
        else:
            solver = None
        
        # Continue with existing implementation...
        # ...
        
        # Solve with specific solver if configured
        if solver:
            prob.solve(solver)
        else:
            prob.solve()
        
        # Continue with existing implementation...
        # ...
    except Exception as e:
        logger.error(f"Error during algorithm execution: {str(e)}")
        raise
```

### Stage 4: Result Analysis Decisions

Result analysis decisions control how results are presented and saved.

```python
# Add to run_allocation_process_interactive in main.py
if current_stage == "result_analysis":
    click.echo(click.style("\nResult Analysis Options:", fg="cyan"))
    
    # Output format options
    output_format = click.prompt(
        "Output format",
        type=click.Choice(['html', 'csv', 'json', 'all']),
        default='html'
    )
    
    # Visualization options
    visualization_type = click.prompt(
        "Visualization type",
        type=click.Choice(['basic', 'detailed', 'comparative']),
        default='detailed'
    )
    
    # Whether to include sensitivity analysis
    sensitivity_analysis = click.confirm(
        "Include sensitivity analysis?",
        default=False
    )
    
    analysis_params = {
        'output_format': output_format,
        'visualization_type': visualization_type,
        'sensitivity_analysis': sensitivity_analysis
    }
    
    # Pass the params to the stage execution
    success = allocation_service.execute_stage(
        current_stage,
        algorithm_params=analysis_params
    )
```

#### Implementation in AllocationService

```python
def _execute_result_analysis(self) -> bool:
    """Execute result analysis with decision parameters"""
    try:
        # Get decision parameters if available
        params = self.stage_handler.stages.get("result_analysis", {}).get("decisions", {})
        output_format = params.get("output_format", "html")
        visualization_type = params.get("visualization_type", "detailed")
        sensitivity_analysis = params.get("sensitivity_analysis", False)
        
        # Log decisions
        logger.info(f"Result analysis parameters: format={output_format}, " +
                    f"visualization={visualization_type}, sensitivity={sensitivity_analysis}")
        
        # Track progress
        if self.stage_handler:
            self.stage_handler.track_progress(
                "result_analysis", 
                0.1, 
                "Starting result analysis"
            )
        
        # For each algorithm that was run, save its output
        for algorithm_name, algorithm in self.algorithm_results.items():
            # Track progress for each algorithm
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "result_analysis", 
                    0.3, 
                    f"Processing output for {algorithm_name}"
                )
            
            if hasattr(algorithm, 'save_output'):
                # Pass format parameter to save_output if supported
                algorithm.save_output(output_format=output_format)
                logger.info(f"Saved output for algorithm: {algorithm_name} in format: {output_format}")
        
        # Generate visualizations based on selected type
        if len(self.algorithm_results) > 0:
            # Track progress
            if self.stage_handler:
                self.stage_handler.track_progress(
                    "result_analysis", 
                    0.5, 
                    f"Generating {visualization_type} visualizations"
                )
            
            # Import visualizer
            try:
                from src.visualization.result_visualizer import ResultVisualizer
                
                # Create visualizer
                visualizer = ResultVisualizer()
                
                # Generate report with specified visualization type
                process_summary = self.get_process_summary() if self.stage_handler else None
                report_path = visualizer.generate_html_report(
                    self.algorithm_results, 
                    process_summary,
                    visualization_type=visualization_type
                )
                
                # Perform sensitivity analysis if requested
                if sensitivity_analysis:
                    if self.stage_handler:
                        self.stage_handler.track_progress(
                            "result_analysis", 
                            0.7, 
                            "Performing sensitivity analysis"
                        )
                    
                    # Example of sensitivity analysis
                    self._perform_sensitivity_analysis()
                
                # Track report generation
                if report_path and report_path != "Report generation failed":
                    if self.stage_handler:
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
        if self.stage_handler:
            self.stage_handler.track_progress(
                "result_analysis", 
                1.0, 
                "Result analysis complete"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error in result analysis: {str(e)}")
        if self.stage_handler:
            self.stage_handler.track_progress(
                "result_analysis", 
                0.0, 
                f"Error in result analysis: {str(e)}"
            )
        return False
    
def _perform_sensitivity_analysis(self):
    """Perform sensitivity analysis on the results"""
    logger.info("Performing sensitivity analysis")
    
    # Implementation depends on your specific needs
    # This is just a placeholder for the concept
    
    for algorithm_name, result in self.algorithm_results.items():
        if algorithm_name == "fillbags":
            # Analyze how changing capacity would affect results
            logger.info(f"Analyzing sensitivity for {algorithm_name}")
            
            # Example analysis (would need to be implemented)
            # result.analyze_capacity_sensitivity()
            
        elif algorithm_name == "lp":
            # Analyze how changing weights would affect results
            logger.info(f"Analyzing sensitivity for {algorithm_name}")
            
            # Example analysis (would need to be implemented)
            # result.analyze_weight_sensitivity()
    
    logger.info("Sensitivity analysis complete")
```

## Modifying the Process Stage Handler

Update the `ProcessStageHandler` to better handle decision recording by enhancing the `record_stage_decision` method:

```python
def record_stage_decision(self, stage_name: str, algorithm_name: str = None, parameters: Dict[str, Any] = None) -> None:
    """
    Record a decision made for a stage
    
    Args:
        stage_name: Name of the stage
        algorithm_name: Optional name of the algorithm (for resource_allocation stage)
        parameters: Decision parameters
    """
    if stage_name not in self.stages:
        raise InvalidStageSequenceError(f"Unknown stage: {stage_name}")
        
    stage = self.stages[stage_name]
    
    # Record the decision differently based on stage type
    if algorithm_name:
        # For stages with algorithms like resource_allocation
        stage['decisions'][algorithm_name] = parameters
        logger.info(f"Recorded algorithm decision for stage {stage_name}: {algorithm_name} with {len(parameters)} parameters")
    else:
        # For other stages, use the stage name as key
        stage['decisions'][stage_name] = parameters
        logger.info(f"Recorded decision for stage {stage_name} with {len(parameters) if parameters else 0} parameters")
    
    # Record in process manager if available
    if self.process_manager:
        # Get stage sequence number
        sequence = stage['sequence']
        
        try:
            # Make the decision in the process manager
            self.process_manager.make_decisions(
                stage=sequence,
                decision_values={
                    'algorithm': algorithm_name,
                    'parameters': parameters
                }
            )
            
            logger.info(f"Stored decision in process manager for stage {stage_name}")
            
        except Exception as e:
            logger.error(f"Error recording decision: {str(e)}")
            # Continue execution even if decision recording fails
```

## Updating the Interactive Process

Modify the `run_allocation_process_interactive` function to record decisions for each stage:

```python
# Add to run_allocation_process_interactive after getting user decisions for a stage
# For example, after getting data_loading decisions:

# Record the decision
allocation_service.stage_handler.record_stage_decision(
    stage_name=current_stage,
    parameters=data_params  # The parameters collected from user
)
```

Make sure to add this for each stage where you collect decisions.

## Testing Your Implementation

Test your implementation with these steps:

1. Run the interactive process:
   ```bash
   python main.py run
   ```

2. At each stage, make decisions and observe:
   - Are your decisions being applied correctly?
   - Is the process properly recording decisions?
   - Can you go back and change decisions?
   - Do the results reflect your decisions?

3. Check logs to verify decision recording:
   ```
   grep "decision" logs/bag_allocation_algo_*.log
   ```

## Extending the System

Consider these extensions once your decision points are working:

1. **Decision Templates**: Allow saving and loading sets of decisions for reuse
   ```python
   # Add commands to save/load decision templates
   @click.command(help="Save current decisions as a template")
   @click.argument("template_name")
   def save_template(template_name):
       # Implementation to save current decisions as a template
       pass
   
   @click.command(help="Load decisions from a template")
   @click.argument("template_name")
   def load_template(template_name):
       # Implementation to load decisions from a template
       pass
   ```

2. **Automated Decision Analysis**: Analyze the impact of different decisions
   ```python
   # Add a command to analyze decision impacts
   @click.command(help="Analyze impact of different decisions")
   @click.option("--iterations", default=10, help="Number of iterations to run")
   def analyze_decisions(iterations):
       # Implementation to run process with different decisions and compare results
       pass
   ```

3. **Decision Visualization**: Create visualizations showing how decisions affected the outcome
   ```python
   # Add to result visualization
   def visualize_decision_impact(self, decisions, results):
       # Implementation to show correlation between decisions and outcomes
       pass
   ```

By implementing these decision points throughout your system, you'll create a powerful interactive workflow that gives users full control over the resource allocation process while maintaining the ability to track and reproduce results.
