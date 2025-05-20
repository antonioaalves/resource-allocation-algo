"""Batch processing module for running resource allocation without user interaction"""

# Dependencies
import logging
import click
import time
import os
from typing import Dict, Any
from datetime import datetime
from base_data_project.utils import create_components

# Import project stuff
from src.config import CONFIG
from src.services.allocation_service import AllocationService
from src.diagnostics.process_diagnostics import ProcessDiagnostics

# Get logger
logger = logging.getLogger('BagAllocationAlgo')

def run_allocation_process(data_manager, process_manager):
    """Logic for running the allocation process as a whole without user interaction"""
    logger.info("Starting batch allocation process")
    
    try:
        # Create the allocation service with data and process managers
        allocation_service = AllocationService(
            data_manager=data_manager,
            process_manager=process_manager
        )
        
        # Initialize a new process
        process_id = allocation_service.initialize_process(
            "Resource Allocation Batch Run", 
            f"Resource allocation process run on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        logger.info(f"Initialized process with ID: {process_id}")
        
        # Display process start
        click.echo(click.style(f"Starting resource allocation process (ID: {process_id})", fg="green", bold=True))
        click.echo()
        
        # Execute the data loading stage
        click.echo(click.style("Stage 1: Loading data...", fg="blue"))
        success = allocation_service.execute_stage("data_loading")
        
        if not success:
            click.echo(click.style("✘ Data loading failed", fg="red", bold=True))
            return False
        else:
            click.echo(click.style("✓ Data loading completed successfully", fg="green"))
            click.echo()
        
        # Execute the data transformation stage
        click.echo(click.style("Stage 2: Transforming data...", fg="blue"))
        success = allocation_service.execute_stage("data_transformation")
        
        if not success:
            click.echo(click.style("✘ Data transformation failed", fg="red", bold=True))
            return False
        else:
            click.echo(click.style("✓ Data transformation completed successfully", fg="green"))
            click.echo()
            
        # Execute resource allocation using the configured algorithms
        click.echo(click.style("Stage 3: Resource allocation...", fg="blue"))
        algorithms = CONFIG.get('stages', {}).get('resource_allocation', {}).get('algorithms', ['fillbags'])
        
        all_success = True
        for algorithm in algorithms:
            click.echo(f"  Running allocation algorithm: {algorithm}")
            algorithm_params = CONFIG.get('algorithm_defaults', {}).get(algorithm, {})
            
            success = allocation_service.execute_stage(
                "resource_allocation", 
                algorithm_name=algorithm,
                algorithm_params=algorithm_params
            )
            
            if not success:
                click.echo(click.style(f"  ✘ Algorithm '{algorithm}' failed", fg="red"))
                all_success = False
            else:
                click.echo(click.style(f"  ✓ Algorithm '{algorithm}' completed successfully", fg="green"))
        
        if not all_success:
            click.echo(click.style("⚠ Some allocation algorithms failed", fg="yellow", bold=True))
        else:
            click.echo(click.style("✓ All allocation algorithms completed successfully", fg="green"))
        click.echo()
        
        # Execute result analysis
        click.echo(click.style("Stage 4: Analyzing results...", fg="blue"))
        success = allocation_service.execute_stage("result_analysis")
        
        if not success:
            click.echo(click.style("✘ Result analysis failed", fg="red", bold=True))
            all_success = False
        else:
            click.echo(click.style("✓ Result analysis completed successfully", fg="green"))
        click.echo()
        
        # Finalize the process
        allocation_service.finalize_process()
        
        # Capture diagnostics if process manager is available
        if process_manager:
            process_summary = allocation_service.get_process_summary()
            
            try:
                # Initialize diagnostics
                diagnostics = ProcessDiagnostics()
                
                # Capture process snapshot
                snapshot_path = diagnostics.capture_process_snapshot(process_summary)
                
                # Generate performance report
                report_path = diagnostics.generate_performance_report(process_summary)
                
                if report_path:
                    click.echo(click.style("\nPerformance Analysis:", fg="blue", bold=True))
                    click.echo(f"A performance report has been generated at:")
                    click.echo(click.style(f"{report_path}", fg="green"))
                    
                logger.info("Process diagnostics generated successfully")
                
            except Exception as e:
                logger.warning(f"Failed to generate process diagnostics: {str(e)}")
        
        # Display process summary
        if process_manager:
            process_summary = allocation_service.get_process_summary()
            status_counts = process_summary.get('status_counts', {})
            
            click.echo(click.style("Process Summary:", fg="blue", bold=True))
            click.echo(f"Process ID: {process_id}")
            click.echo(f"Completed stages: {status_counts.get('completed', 0)}")
            click.echo(f"Failed stages: {status_counts.get('failed', 0)}")
            click.echo(f"Overall progress: {process_summary.get('progress', 0) * 100:.1f}%")
            click.echo()
        
        # Display output location
        output_dir = os.path.abspath("data/output")
        visualization_dir = os.path.abspath("data/output/visualizations")
        
        click.echo(click.style("Output Files:", fg="blue", bold=True))
        click.echo(f"Raw results have been saved to: {output_dir}")
        
        # Check if HTML report was generated
        if success:
            html_report = None
            # Look for the most recent HTML report
            try:
                report_files = [f for f in os.listdir(visualization_dir) if f.endswith('.html')]
                if report_files:
                    # Sort by modification time, newest first
                    report_files.sort(key=lambda x: os.path.getmtime(os.path.join(visualization_dir, x)), reverse=True)
                    html_report = os.path.join(visualization_dir, report_files[0])
            except Exception:
                pass
                
            if html_report and os.path.exists(html_report):
                click.echo(click.style("\nVisualization Report:", fg="blue", bold=True))
                click.echo(f"An HTML report with visualizations has been generated at:")
                click.echo(click.style(f"{html_report}", fg="green"))
                click.echo(f"\nOpen this file in a web browser to view detailed results and visualizations.")
        
        click.echo()
        
        return all_success
        
    except Exception as e:
        logger.error(f"Error in allocation process: {str(e)}", exc_info=True)
        click.echo(click.style(f"Error in allocation process: {str(e)}", fg="red", bold=True))
        return False

@click.command(help="Run the resource allocation process in batch mode (non-interactive)")
@click.option("--use-db/--use-csv", prompt="Use database for data storage", default=False, 
              help="Use database instead of CSV files")
@click.option("--no-tracking/--enable-tracking", default=False, 
              help='Disable process tracking (reduces overhead)')
@click.option("--algorithm", "-a", type=click.Choice(['fillbags', 'lp', 'both']), default='both',
              help="Select which allocation algorithm to use")
def new_process(use_db, no_tracking, algorithm):
    """
    Batch process run with enhanced user experience (non-interactive)
    """
    # This function now lives in batch_process.py
    # It's imported into main.py and registered as the 'batch' command
    
    # Display header
    click.clear()
    click.echo(click.style("=== Resource Allocation System (Batch Mode) ===", fg="green", bold=True))
    click.echo(click.style(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fg="green"))
    click.echo()
    
    # Override algorithm configuration if specified
    if algorithm != 'both':
        CONFIG['stages']['resource_allocation']['algorithms'] = [algorithm]
    
    # Display configuration
    click.echo(click.style("Configuration:", fg="blue"))
    click.echo(f"Data source: {'Database' if use_db else 'CSV files'}")
    click.echo(f"Process tracking: {'Disabled' if no_tracking else 'Enabled'}")
    click.echo(f"Algorithms: {', '.join(CONFIG['stages']['resource_allocation']['algorithms'])}")
    click.echo()
    
    try:
        logger.info("Starting the Batch Allocation Process")
        click.echo("Initializing components...")
        
        # Create spinner for initialization
        with click.progressbar(length=100, label="Initializing") as bar:
            # Create and configure components
            data_manager, process_manager = create_components(use_db, no_tracking)
            bar.update(100)
        
        click.echo()
        click.echo(click.style("Components initialized successfully", fg="green"))
        click.echo()
        
        start_time = time.time()
        
        with data_manager:
            # Run the allocation process
            success = run_allocation_process(data_manager=data_manager, process_manager=process_manager)

            # Log final status
            if success:
                logger.info("Application completed successfully")
                click.echo(click.style("\n✓ Application completed successfully", fg="green", bold=True))
            else:
                logger.warning("Application completed with errors")
                click.echo(click.style("\n⚠ Application completed with errors", fg="yellow", bold=True))
                
        # Display execution time
        execution_time = time.time() - start_time
        click.echo(f"Total execution time: {execution_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        click.echo(click.style(f"\n✘ Application failed: {str(e)}", fg="red", bold=True))
        raise
    finally:
        logger.info("Application finished")