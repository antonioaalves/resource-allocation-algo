"""Project Main script with enhanced CLI and interactive allocation process"""

# Dependencies
import logging
import os
from base_data_project.log_config import setup_logger
from src.helpers import reconfigure_all_loggers

# 1. Create logs directory
os.makedirs('logs', exist_ok=True)

# 2. Set up your project logger
logger = setup_logger(
    project_name='resource_allocation_algo',
    log_level=logging.DEBUG,
    log_dir='logs',
    console_output=True
)

# Run the reconfiguration
reconfigure_all_loggers(logger)

# 4. Also preemptively fix the framework's main logger
framework_logger = logging.getLogger('base_data_project')
framework_logger.setLevel(logging.DEBUG)
# Clear existing handlers
for handler in framework_logger.handlers[:]:
    framework_logger.removeHandler(handler)
# Add your handlers
for handler in logger.handlers:
    framework_logger.addHandler(handler)

logger.info("Main script starting - logging configured. Starting importing dependencies")

import click
import time
import sys
from typing import Dict, Any
from datetime import datetime
from base_data_project.data_manager.factory import DataManagerFactory
from base_data_project.process_management.manager import ProcessManager
from base_data_project.utils import create_components
from base_data_project.data_manager.managers.managers import BaseDataManager
from base_data_project.process_management.manager import ProcessManager

# Import project stuff
from batch_process import new_process  # Import the batch process command
from src.config import CONFIG, PROJECT_NAME
from src.helpers import parse_allocations
from src.services.allocation_service import AllocationService
from src.diagnostics.process_diagnostics import ProcessDiagnostics


# Define the click group
@click.group()
def cli():
    """Resource Allocation System - Command Line Interface"""
    pass

def run_allocation_process_interactive(data_manager: BaseDataManager, process_manager: ProcessManager):
    """Logic for running the allocation process with interactive user decisions"""
    logger.info("Starting interactive allocation process")
    
    try:

        # Create the allocation service with data and process managers
        allocation_service = AllocationService(
            data_manager=data_manager,
            process_manager=process_manager
        )
        
        # Initialize a new process
        process_id = allocation_service.initialize_process(
            "Interactive Resource Allocation Run", 
            f"Resource allocation process run on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        logger.info(f"Initialized process with ID: {process_id}")
        
        # Display process start
        click.echo(click.style(f"Starting interactive resource allocation process (ID: {process_id})", fg="green", bold=True))
        click.echo()
        
        # Get stages from CONFIG and sort by sequence
        stages = list(CONFIG.get('stages', {}).keys())
        stages.sort(key=lambda stage: CONFIG['stages'][stage].get('sequence', 0))
        
        # Track the current stage index
        current_stage_idx = 0
        stage_results = {}
        
        # Loop until process is complete or user exits
        while 0 <= current_stage_idx < len(stages):
            current_stage = stages[current_stage_idx]
            stage_config = CONFIG['stages'][current_stage]
            
            # Display current stage
            click.echo(click.style(f"\nCurrent Stage: {current_stage.replace('_', ' ').title()} (Stage {current_stage_idx + 1}/{len(stages)})", fg="blue", bold=True))
            
            # Check if stage already executed
            if current_stage in stage_results:
                click.echo(click.style(f"This stage has already been executed with result: {'Success' if stage_results[current_stage] else 'Failed'}", fg="yellow"))
                redo = click.confirm("Do you want to re-execute this stage?", default=False)
                if not redo:
                    # Ask to go forward or backward
                    direction = click.prompt(
                        "Enter 'n' for next stage, 'p' for previous stage, or 'q' to quit",
                        type=click.Choice(['n', 'p', 'q']),
                        default='n'
                    )
                    
                    if direction == 'n':
                        current_stage_idx += 1
                        continue
                    elif direction == 'p':
                        current_stage_idx -= 1
                        continue
                    else:  
                        # 'q'
                        break

            if 'decisions' in stage_config:
                click.echo(click.style("\nStage Configuration Options:", fg="cyan"))

                # Process each decision in this stage
                for decision_name, defaults in stage_config['decisions'].items():
                    # It formats the decision name (replacing underscores with spaces and capitalizing) and displays it to the user with cyan styling.
                    click.echo(click.style(f"\n{decision_name.replace('_', ' ').title()}:", fg="cyan"))

                    # Handle different type of decisions
                    if decision_name == 'selections' and current_stage == 'data_loading':
                        loading_options = {}

                        # Prompt the user for the list of months
                        choosen_months = click.prompt(
                            text="List of months to plan (comma-separated)",
                            type=str,
                            default=defaults.get('months', [1]),
                            show_default=True                            
                        )
                        if choosen_months.strip():
                            loading_options['months'] = [int(t.strip()) for t in choosen_months.split(',')]

                        choosen_years = click.prompt(
                            text="List of years to plan (comma-separated)",
                            type=str,
                            default=defaults.get('years', [2024]),
                            show_default=True                            
                        )
                        if choosen_years.strip():
                            loading_options['years'] = [int(t.strip()) for t in choosen_years.split(',')]

                        apply_selection = True
                        if process_manager:
                            process_manager.make_decisions(
                                stage=stage_config['sequence'],
                                decision_values={
                                    'selections': {
                                        'apply_selection': apply_selection,
                                        **loading_options
                                    }
                                }
                            )

                    elif decision_name == 'filtering' and current_stage == 'data_transformation':
                        # Let the user decide if he wants to take out any colabs
                        apply_filtering = click.confirm(
                            text="Apply data filtering?",
                            default=defaults.get("apply_filtering", True)
                        )
                        filter_options = {}
                        if apply_filtering:
                            # Prompt for which colabs to exclude
                            excluded_workers = click.prompt(
                                text="List of excluded employees (comma-separated, leave empty to skip)",
                                type=str,
                                default=defaults.get('excluded_employees'),
                                show_default=True
                            )
                            if len(excluded_workers.strip()) == 0: 
                                filter_options['filter_by_employees'] = ''
                            elif excluded_workers.strip():
                                filter_options['filter_by_employees'] = [t.strip() for t in excluded_workers.split(',')]
                            print(f"filter_options['filter_by_employees']: {filter_options['filter_by_employees']}")

                            # Prompt for production lines to exclude
                            excluded_lines = click.prompt(
                                text="List of excluded production lines (comma-separated, leave empty to skip)",
                                type=str,
                                default=defaults.get('excluded_lines'),
                                show_default=True
                            )
                            if len(excluded_lines.strip()) == 0: 
                                filter_options['filter_by_lines'] = ''
                            elif excluded_lines.strip():
                                filter_options['filter_by_lines'] = [t.strip() for t in excluded_lines.split(',')]
                            print(f"filter_options['filter_by_lines']: {filter_options['filter_by_lines']}")

                            logger.info(f"Excluded employees being stored: {filter_options['filter_by_employees']}")
                            logger.info(f"Excluded lines being stored: {filter_options['filter_by_lines']}")    
                        
                        if process_manager:
                            # Create the filtering decision properly structured
                            filtering_decision = {
                                'filtering': {  # This matches the key expected in allocation_service.py
                                    'apply_filtering': apply_filtering,
                                    **filter_options  # This includes excluded_employees and excluded_lines
                                }
                            }
                            
                            # Log the complete filtering decision before storing
                            logger.info(f"Storing filtering decision: {filtering_decision}")
                            
                            # Make the decision in the process manager
                            process_manager.make_decisions(
                                stage=stage_config['sequence'],
                                decision_values=filtering_decision
                            )

                    elif decision_name == 'product_assignments' and current_stage == 'product_allocation':
                        if hasattr(allocation_service.data, 'printable_df'):
                            printable_df = allocation_service.data.printable_df.copy()
                            
                            # Add some informative headers
                            click.echo(click.style("\nAvailable Data Preview (product_production_lines information):", fg="cyan"))
                            click.echo(click.style("This shows filtered data based on your previous filtering decisions", fg="cyan"))
                            
                            # Format the DataFrame for better display
                            # Limit columns for clearer output
                            display_cols = ['product_id', 'production_line_id', 'product_name', 'production_line_name', 'month', 'year', 'real_hours_amount', 'operating_type_id', 'theoretical_hours_amount', 'delta_hours_amount']
                            
                            if set(display_cols).issubset(set(printable_df.columns)):
                                display_df = printable_df[display_cols].copy()
                                
                                # Round numeric columns to improve readability
                                for col in ['real_hours_amount', 'delta_hours_amount']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].round(2)
                                
                                # Check the DataFrame size and potentially limit rows if too large
                                if len(display_df) > 20:
                                    click.echo(click.style(f"Showing first 20 rows of {len(display_df)} total", fg="yellow"))
                                    click.echo(display_df.head(20))
                                    click.echo(click.style(f"... {len(display_df) - 20} more rows not shown ...", fg="yellow"))
                                else:
                                    click.echo(display_df)
                            else:
                                # Fall back to standard display if columns don't match
                                click.echo(printable_df)
                        else:
                            click.echo(click.style("\nProduct production line data not available yet.", fg="yellow"))
                        
                        # Improved explanation for how to enter product allocations
                        click.echo(click.style("\nProduct Allocation Input Guide:", fg="cyan"))
                        click.echo("Format: 'product_id:production_line_id:operating_type:quantity,product_id:production_line_id:quantity,...'")
                        click.echo("Example: '1:14:2:500,AE180:L16:200:10'")
                        click.echo("Leave empty to skip allocations for this run.")
                        
                        # Get the allocations
                        product_allocations = click.prompt(
                            text="Enter product allocations in production lines",
                            type=str,
                            default=defaults.get('product_assignments'),
                            show_default=True
                        )
                        
                        # Parse the allocations
                        allocations_dict = parse_allocations(product_allocations)
                        print(allocations_dict)
                        
                        # Show a confirmation of what was entered
                        if allocations_dict:
                            click.echo(click.style("\nConfirmed allocations:", fg="green"))
                            for product_id, allocation in allocations_dict.items():
                                click.echo(f"  Product {product_id} → Line {allocation['target']}: {allocation['quantity']} units")
                        
                        if process_manager:
                            process_manager.make_decisions(
                                stage=stage_config['sequence'],
                                decision_values={
                                    'product_assignments': allocations_dict
                                }
                            )

                    elif decision_name == 'time_periods' and current_stage == 'data_transformation':
                        filter_options = {}
                        # Check if defaults is a dictionary or a direct value
                        default_time_periods = defaults
                        if isinstance(defaults, dict):
                            default_time_periods = defaults.get('time_periods')
                        
                        # Prompt for time period
                        time_periods = click.prompt(
                            text='Define the number of time period for the algorithmic calculations',
                            type=int,
                            default=default_time_periods,
                            show_default=True
                        )
                        
                        if time_periods:
                            filter_options['time_periods'] = time_periods

                        if process_manager:
                            process_manager.make_decisions(
                                stage=stage_config['sequence'],
                                decision_values={
                                    'time_periods': time_periods
                                }
                            )

                    elif decision_name == 'changes' and current_stage == 'result_analysis':
                        add_changes = click.confirm(
                            text="Want to manually change any allocation?",
                            default=defaults.get("add_changes", False)
                        )
                        changes = {}

                        if add_changes:
                            # Prompt the user for which changes 
                            allocation_changes = click.prompt(
                                text="Define which employees should manually allocated to a different production line (comma-separated employee_id:prodline_id, example -> 3:L27,2:L15)",
                                default=defaults.get('special_allocations', {})
                            )
                            if allocation_changes:
                                changes['allocation_changes'] = [t.strip() for t in allocation_changes]

                        if process_manager:
                            process_manager.make_decisions(
                                stage=stage_config['sequence'],
                                decision_values={
                                    'changes': {
                                        'add_changes': add_changes,
                                        **changes
                                    }
                                }
                            )
                    
                    elif decision_name == 'generate_report' and current_stage == 'result_analysis':
                        generate_report = click.confirm(
                            text="Want to generate the output report in html?",
                            default=defaults.get('generate_report', False)
                        )

                        if process_manager:
                            process_manager.make_decisions(
                                stage=current_stage,
                                decision_values={
                                    'generate_report': generate_report
                                }
                            )
                
          

            # For resource allocation stage, prompt for algorithm choice
            algorithm_name = None
            algorithm_params = {}
            
            if current_stage == "resource_allocation":
                click.echo(click.style("\nResource Allocation Options:", fg="cyan"))
                
                # Get algorithm choices from config
                available_algorithms = CONFIG.get('stages', {}).get('resource_allocation', {}).get('algorithms', ['fillbags'])
                
                # Prompt user for algorithm choice
                algorithm_name = click.prompt(
                    "Choose algorithm",
                    type=click.Choice(available_algorithms),
                    default=available_algorithms[0]
                )
                
                # Get default parameters
                default_params = CONFIG.get('algorithm_defaults', {}).get(algorithm_name, {})
                
                # For FillBags algorithm, prompt for specific parameters
                if algorithm_name == "fillbags":
                    click.echo(click.style("\nFillBags Algorithm Parameters:", fg="cyan"))
                    
                    sort_strategy = click.prompt(
                        "Sort strategy",
                        type=click.Choice(['by_colors', 'by_capacity', 'random']),
                        default=default_params.get('sort_strategy', 'by_colors')
                    )
                    
                    prioritize_high = click.confirm(
                        "Prioritize high capacity?",
                        default=default_params.get('prioritize_high_capacity', True)
                    )
                    
                    algorithm_params = {
                        'sort_strategy': sort_strategy,
                        'prioritize_high_capacity': prioritize_high
                    }
                
                # For LP algorithm, prompt for specific parameters
                elif algorithm_name == "lp":
                    click.echo(click.style("\nLP Algorithm Parameters:", fg="cyan"))
                    
                    temporal_space = click.prompt(
                        "Temporal space",
                        type=int,
                        default=default_params.get('temporal_space', 1)
                    )
                    
                    understaffing_weight = click.prompt(
                        "Understaffing weight",
                        type=float,
                        default=default_params.get('objective_weights', {}).get('understaffing', 1.0)
                    )
                    
                    overstaffing_weight = click.prompt(
                        "Overstaffing weight",
                        type=float,
                        default=default_params.get('objective_weights', {}).get('overstaffing', 1.0)
                    )
                    
                    algorithm_params = {
                        'temporal_space': temporal_space,
                        'objective_weights': {
                            'understaffing': understaffing_weight,
                            'overstaffing': overstaffing_weight
                        }
                    }

                if process_manager:
                    process_manager.make_decisions(
                        stage=stage_config['sequence'],
                        decision_values={
                            'algorithm': {
                                'name': algorithm_name,
                                'parameters': algorithm_params
                            }
                        }
                    )

            # Execute the current stage
            click.echo(click.style(f"\nExecuting {current_stage}...", fg="blue"))
            
            with click.progressbar(length=100, label="Processing") as bar:
                # Simulate progress
                for i in range(10, 91, 20):
                    time.sleep(0.2)  # Simulate processing time
                    bar.update(i)
                
                # Execute the stage
                if current_stage == "resource_allocation":
                    success = allocation_service.execute_stage(current_stage, algorithm_name, algorithm_params)
                else:
                    success = allocation_service.execute_stage(current_stage)
                
                bar.update(100 - bar.pos)  # Complete the progress bar
            
            # Store the result
            stage_results[current_stage] = success
            
            # Display result
            if success:
                click.echo(click.style(f"\n✓ {current_stage.replace('_', ' ').title()} completed successfully", fg="green"))
            else:
                click.echo(click.style(f"\n✘ {current_stage.replace('_', ' ').title()} failed", fg="red", bold=True))
            
            # After executing, prompt for next action
            direction = click.prompt(
                "\nEnter 'n' for next stage, 'p' for previous stage, or 'q' to quit",
                type=click.Choice(['n', 'p', 'q']),
                default='n'
            )
            
            if direction == 'n':
                current_stage_idx += 1
            elif direction == 'p':
                current_stage_idx -= 1
            else:  # 'q'
                break
        
        # Finalize the process
        allocation_service.finalize_process()
        process_summary = allocation_service.get_process_summary()
        
        # Display process summary
        click.echo(click.style("\nProcess Summary:", fg="blue", bold=True))
        click.echo(f"Process ID: {process_id}")
        click.echo(f"Completed stages: {process_summary.get('status_counts', {}).get('completed', 0)}")
        click.echo(f"Failed stages: {process_summary.get('status_counts', {}).get('failed', 0)}")
        click.echo(f"Overall progress: {process_summary.get('progress', 0) * 100:.1f}%")
        
        # Return overall success status
        all_success = all(stage_results.values())
        return all_success
    
    except click.Abort:
        logger.warning("User aborted the interactive process")
        click.echo(click.style("\n⚠ Process aborted by user", fg="yellow", bold=True))
        return 1  # Return error code
                
    except Exception as e:
        logger.error(f"Error in interactive allocation process: {str(e)}", exc_info=True)
        click.echo(click.style(f"\n✘ Process failed: {str(e)}", fg="red", bold=True))
        raise


@click.command(help="Run an interactive resource allocation process")
@click.option("--use-db/--use-csv", prompt="Use database for data storage", default=False, 
              help="Use database instead of CSV files")
@click.option("--no-tracking/--enable-tracking", default=False, 
              help='Disable process tracking (reduces overhead)')
def interactive_process(use_db, no_tracking):
    """
    Interactive process run with user decision points
    """
    # Display header
    click.clear()
    click.echo(click.style("=== Interactive Resource Allocation System ===", fg="green", bold=True))
    click.echo(click.style(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fg="green"))
    click.echo()
    
    # Display configuration
    click.echo(click.style("Configuration:", fg="blue"))
    click.echo(f"Data source: {'Database' if use_db else 'CSV files'}")
    click.echo(f"Process tracking: {'Disabled' if no_tracking else 'Enabled'}")
    click.echo()
    
    try:
        logger.info("Starting the Interactive Allocation Process")
        click.echo("Initializing components...")
        
        # Create spinner for initialization
        with click.progressbar(length=100, label="Initializing") as bar:
            # Create and configure components
            data_manager, process_manager = create_components(use_db, no_tracking, config=CONFIG)
            bar.update(100)
        
        click.echo()
        click.echo(click.style("Components initialized successfully", fg="green"))
        click.echo()
        
        start_time = time.time()
        
        with data_manager:
            # Run the interactive allocation process
            success = run_allocation_process_interactive(data_manager=data_manager, process_manager=process_manager)

            # Log final status
            if success:
                logger.info("Interactive process completed successfully")
                click.echo(click.style("\n✓ Process completed successfully", fg="green", bold=True))
            else:
                logger.warning("Interactive process completed with some failures")
                click.echo(click.style("\n⚠ Process completed with some failures", fg="yellow", bold=True))
                
        # Display execution time
        execution_time = time.time() - start_time
        click.echo(f"Total execution time: {execution_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Interactive process failed: {str(e)}", exc_info=True)
        click.echo(click.style(f"\n✘ Process failed: {str(e)}", fg="red", bold=True))
        raise
    finally:
        logger.info("Interactive process finished")


@click.command(help="Display system information")
def info():
    """Display system information and configuration"""
    # Display header
    click.clear()
    click.echo(click.style("=== Resource Allocation System Information ===", fg="green", bold=True))
    click.echo()
    
    # Display version and configuration
    click.echo(click.style("System Version:", fg="blue", bold=True))
    click.echo("Version: 1.0.0")
    click.echo(f"Python executable: {sys.executable}")
    click.echo()
    
    # Display available algorithms
    click.echo(click.style("Available Algorithms:", fg="blue", bold=True))
    for algo_name, algo_config in CONFIG.get('algorithm_defaults', {}).items():
        click.echo(f"- {algo_name}")
        for param, value in algo_config.items():
            click.echo(f"  {param}: {value}")
    click.echo()
    
    # Display data source configuration
    click.echo(click.style("Data Sources:", fg="blue", bold=True))
    click.echo(f"Default mode: {'Database' if CONFIG.get('use_db', False) else 'CSV'}")
    if not CONFIG.get('use_db', False):
        click.echo("CSV Files:")
        for name, path in CONFIG.get('dummy_data_filepaths', {}).items():
            exists = os.path.exists(path)
            status = click.style("✓", fg="green") if exists else click.style("✘", fg="red")
            click.echo(f"  {name}: {path} {status}")
    else:
        click.echo(f"Database URL: {CONFIG.get('db_url', 'not configured')}")
    click.echo()
    
    # Display log configuration
    click.echo(click.style("Logging:", fg="blue", bold=True))
    click.echo(f"Log level: {CONFIG.get('log_level', 'INFO')}")
    log_dir = os.path.abspath("logs")
    click.echo(f"Log directory: {log_dir}")
    click.echo()


@click.command(help="Analyze process performance")
@click.option("--snapshots", "-s", multiple=True, help="Paths to process snapshot files to compare")
@click.option("--latest", "-l", is_flag=True, help="Use the most recent snapshots")
@click.option("--count", "-c", default=3, help="Number of recent snapshots to compare")
def analyze(snapshots, latest, count):
    """
    Analyze and compare process executions
    """
    click.echo(click.style("=== Process Performance Analysis ===", fg="green", bold=True))
    
    try:
        diagnostics = ProcessDiagnostics()
        
        if not snapshots and not latest:
            # If no specific snapshots provided, just analyze the most recent
            latest = True
            count = 1
        
        snapshot_files = []
        
        if latest:
            # Find the most recent snapshot files
            diagnostics_dir = os.path.abspath("data/diagnostics")
            if os.path.exists(diagnostics_dir):
                snapshot_files = [
                    os.path.join(diagnostics_dir, f)
                    for f in os.listdir(diagnostics_dir)
                    if f.startswith("process_snapshot_") and f.endswith(".json")
                ]
                
                if snapshot_files:
                    # Sort by modification time, newest first
                    snapshot_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    # Take requested number
                    snapshot_files = snapshot_files[:count]
                    
                    click.echo(f"Using {len(snapshot_files)} most recent process snapshots")
                else:
                    click.echo(click.style("No process snapshots found", fg="yellow"))
                    return
            else:
                click.echo(click.style(f"Diagnostics directory not found: {diagnostics_dir}", fg="red"))
                return
        else:
            # Use provided snapshot files
            snapshot_files = list(snapshots)
            
            # Validate files exist
            for file in snapshot_files:
                if not os.path.exists(file):
                    click.echo(click.style(f"Snapshot file not found: {file}", fg="red"))
                    return
        
        # Load the snapshots
        loaded_snapshots = []
        for file in snapshot_files:
            snapshot = ProcessDiagnostics.load_snapshot(file)
            if snapshot:
                loaded_snapshots.append(snapshot)
                click.echo(f"Loaded snapshot: {os.path.basename(file)}")
            else:
                click.echo(click.style(f"Failed to load snapshot: {file}", fg="yellow"))
        
        if not loaded_snapshots:
            click.echo(click.style("No valid snapshots to analyze", fg="red"))
            return
        
        # If only one snapshot, generate a performance report
        if len(loaded_snapshots) == 1:
            click.echo("\nGenerating performance report...")
            report_path = diagnostics.generate_performance_report(loaded_snapshots[0])
            
            if report_path:
                click.echo(click.style("\nPerformance Report:", fg="green"))
                click.echo(f"A detailed performance report has been generated at:")
                click.echo(click.style(f"{report_path}", fg="green"))
            else:
                click.echo(click.style("Failed to generate performance report", fg="red"))
        
        # If multiple snapshots, compare them
        elif len(loaded_snapshots) > 1:
            click.echo("\nComparing process executions...")
            comparison = diagnostics.compare_process_runs(loaded_snapshots)
            
            if "error" in comparison:
                click.echo(click.style(f"Error in comparison: {comparison['error']}", fg="red"))
                return
            
            # Display summary of comparison
            overall = comparison.get("overall_comparison", {})
            success_rate = overall.get("success_rate", 0) * 100
            
            click.echo(click.style("\nComparison Results:", fg="green"))
            click.echo(f"Process runs: {comparison.get('process_count', 0)}")
            click.echo(f"Success rate: {success_rate:.1f}%")
            
            if overall.get("average_completion_time"):
                click.echo(f"Average completion time: {overall['average_completion_time']:.2f} seconds")
                click.echo(f"Fastest run: {overall['fastest_run']:.2f} seconds")
                click.echo(f"Slowest run: {overall['slowest_run']:.2f} seconds")
            
            # Save the comparison
            filepath = diagnostics.save_execution_analysis(comparison)
            if filepath:
                click.echo(f"\nDetailed comparison saved to: {filepath}")
            
    except Exception as e:
        click.echo(click.style(f"Error in analysis: {str(e)}", fg="red"))


# Register commands - make interactive_process the default command
cli.add_command(interactive_process, name="run")  # Make it the run command
cli.add_command(new_process, name="batch")        # Rename to batch for clarity
cli.add_command(info)
cli.add_command(analyze)

if __name__ == "__main__":
    print("Welcome to the Resource Allocation System")
    cli()