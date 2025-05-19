"""
Utility module for process diagnostics and tracking
Provides tools for monitoring and analyzing the allocation process
"""

import logging
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger('BagAllocationAlgo')

class ProcessDiagnostics:
    """
    Utility class for analyzing and diagnosing process execution
    """
    
    def __init__(self, output_dir: str = "data/diagnostics"):
        """
        Initialize diagnostics utility
        
        Args:
            output_dir: Directory for diagnostics output
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"ProcessDiagnostics initialized with output directory: {output_dir}")
    
    def capture_process_snapshot(self, process_summary: Dict[str, Any]) -> str:
        """
        Capture and save a snapshot of the current process state
        
        Args:
            process_summary: Process summary dictionary
            
        Returns:
            Path to the saved snapshot file
        """
        try:
            # Add timestamp to the snapshot
            snapshot = process_summary.copy()
            snapshot['captured_at'] = datetime.now().isoformat()
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"process_snapshot_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, default=str)
                
            logger.info(f"Process snapshot saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error capturing process snapshot: {str(e)}", exc_info=True)
            return None
    
    def analyze_process_execution(self, process_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze process execution and identify issues or bottlenecks
        
        Args:
            process_summary: Process summary dictionary
            
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'status': process_summary.get('status', 'unknown'),
                'issues': [],
                'recommendations': [],
                'stage_analysis': {}
            }
            
            # Analyze stages
            stages = process_summary.get('stages', {})
            failed_stages = []
            slow_stages = []
            
            for stage_name, stage_info in stages.items():
                stage_analysis = {
                    'status': stage_info.get('status'),
                    'issues': [],
                    'metrics': {},
                }
                
                # Check for failed stages
                if stage_info.get('status') == 'failed':
                    failed_stages.append(stage_name)
                    stage_analysis['issues'].append('Stage failed to complete')
                    
                # Check for slow stages (if timing information is available)
                if stage_info.get('started_at') and stage_info.get('completed_at'):
                    started = pd.to_datetime(stage_info['started_at'])
                    completed = pd.to_datetime(stage_info['completed_at'])
                    duration = (completed - started).total_seconds()
                    
                    stage_analysis['metrics']['duration_seconds'] = duration
                    
                    # Arbitrary threshold for demonstration - adjust based on expected performance
                    if duration > 10:  # More than 10 seconds
                        slow_stages.append(stage_name)
                        stage_analysis['issues'].append('Stage execution was slower than expected')
                
                analysis['stage_analysis'][stage_name] = stage_analysis
            
            # Compile overall issues and recommendations
            if failed_stages:
                analysis['issues'].append(f"Failed stages: {', '.join(failed_stages)}")
                analysis['recommendations'].append("Review logs for the failed stages to identify errors")
                
            if slow_stages:
                analysis['issues'].append(f"Slow-performing stages: {', '.join(slow_stages)}")
                analysis['recommendations'].append("Consider optimizing the slow stages or increasing resources")
                
            if not analysis['issues']:
                analysis['issues'].append("No significant issues detected")
                
            if process_summary.get('progress', 0) < 1.0:
                analysis['recommendations'].append("Process did not complete - consider restarting")
                
            logger.info(f"Completed process execution analysis: found {len(analysis['issues'])} issues")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing process execution: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_execution_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Save execution analysis to a file
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            Path to the saved analysis file
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"execution_analysis_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, default=str)
                
            logger.info(f"Execution analysis saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving execution analysis: {str(e)}", exc_info=True)
            return None
    
    def generate_performance_report(self, process_summary: Dict[str, Any], include_recommendations: bool = True) -> str:
        """
        Generate a detailed performance report
        
        Args:
            process_summary: Process summary dictionary
            include_recommendations: Whether to include improvement recommendations
            
        Returns:
            Path to the saved report file
        """
        try:
            # Perform analysis
            analysis = self.analyze_process_execution(process_summary)
            
            # Generate HTML report
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Process Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f0f0f0; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .issue {{ color: #d9534f; }}
                    .recommendation {{ color: #0275d8; }}
                    .success {{ color: #5cb85c; }}
                    .warning {{ color: #f0ad4e; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .footer {{ margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; font-size: 0.8em; color: #777; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Process Performance Report</h1>
                    <p>Report generated on: {timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Process Overview</h2>
                    <div class="summary-box">
                        <p><strong>Process ID:</strong> {process_summary.get('id', 'N/A')}</p>
                        <p><strong>Status:</strong> {process_summary.get('status', 'N/A')}</p>
                        <p><strong>Progress:</strong> {process_summary.get('progress', 0) * 100:.1f}%</p>
                    </div>
                </div>
            """
            
            # Add issues and recommendations section
            if analysis.get('issues') or analysis.get('recommendations'):
                html_content += f"""
                <div class="section">
                    <h2>Analysis Summary</h2>
                """
                
                if analysis.get('issues'):
                    html_content += "<h3>Detected Issues</h3><ul>"
                    for issue in analysis['issues']:
                        html_content += f"<li class='issue'>{issue}</li>"
                    html_content += "</ul>"
                
                if include_recommendations and analysis.get('recommendations'):
                    html_content += "<h3>Recommendations</h3><ul>"
                    for rec in analysis['recommendations']:
                        html_content += f"<li class='recommendation'>{rec}</li>"
                    html_content += "</ul>"
                    
                html_content += "</div>"  # Close section
            
            # Add stage performance details
            html_content += """
                <div class="section">
                    <h2>Stage Performance Details</h2>
                    <table>
                        <tr>
                            <th>Stage</th>
                            <th>Status</th>
                            <th>Execution Time</th>
                            <th>Issues</th>
                        </tr>
            """
            
            # Add rows for each stage
            stages = process_summary.get('stages', {})
            for stage_name, stage_info in stages.items():
                status = stage_info.get('status', 'unknown')
                status_class = 'success' if status == 'completed' else 'warning' if status == 'in_progress' else 'issue'
                
                # Calculate duration if available
                duration = "N/A"
                if stage_info.get('started_at') and stage_info.get('completed_at'):
                    started = pd.to_datetime(stage_info['started_at'])
                    completed = pd.to_datetime(stage_info['completed_at'])
                    duration_secs = (completed - started).total_seconds()
                    duration = f"{duration_secs:.2f} seconds"
                
                # Get issues for this stage
                stage_issues = analysis.get('stage_analysis', {}).get(stage_name, {}).get('issues', [])
                issues_text = ", ".join(stage_issues) if stage_issues else "None"
                
                html_content += f"""
                    <tr>
                        <td>{stage_name}</td>
                        <td class='{status_class}'>{status}</td>
                        <td>{duration}</td>
                        <td class='{"issue" if stage_issues else ""}'>{issues_text}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add footer
            html_content += f"""
                <div class="footer">
                    <p>Resource Allocation System - Performance Report generated on {timestamp}</p>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"performance_report_{timestamp_file}.html"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Performance report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}", exc_info=True)
            return None
    
    def compare_process_runs(self, process_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple process executions
        
        Args:
            process_snapshots: List of process summary dictionaries
            
        Returns:
            Dictionary with comparison results
        """
        try:
            if not process_snapshots or len(process_snapshots) < 2:
                return {"error": "Need at least two process snapshots to compare"}
            
            comparison = {
                "timestamp": datetime.now().isoformat(),
                "process_count": len(process_snapshots),
                "stage_comparisons": {},
                "overall_comparison": {}
            }
            
            # Extract process IDs for reference
            process_ids = [snapshot.get('id', f"Process {i}") for i, snapshot in enumerate(process_snapshots)]
            comparison["process_ids"] = process_ids
            
            # Compare stage performance across processes
            all_stages = set()
            for snapshot in process_snapshots:
                all_stages.update(snapshot.get('stages', {}).keys())
            
            for stage_name in sorted(all_stages):
                stage_comparison = {
                    "status": [],
                    "duration": [],
                    "relative_performance": {}
                }
                
                # Extract stage data
                valid_durations = []
                for i, snapshot in enumerate(process_snapshots):
                    stage_info = snapshot.get('stages', {}).get(stage_name, {})
                    status = stage_info.get('status')
                    stage_comparison["status"].append(status)
                    
                    # Calculate duration if available
                    duration = None
                    if stage_info.get('started_at') and stage_info.get('completed_at') and status == 'completed':
                        started = pd.to_datetime(stage_info['started_at'])
                        completed = pd.to_datetime(stage_info['completed_at'])
                        duration = (completed - started).total_seconds()
                        valid_durations.append(duration)
                    
                    stage_comparison["duration"].append(duration)
                
                # Calculate relative performance if we have valid durations
                if valid_durations:
                    avg_duration = sum(valid_durations) / len(valid_durations)
                    best_duration = min(valid_durations)
                    worst_duration = max(valid_durations)
                    
                    stage_comparison["relative_performance"] = {
                        "average_duration": avg_duration,
                        "best_duration": best_duration,
                        "worst_duration": worst_duration,
                        "variation": (worst_duration - best_duration) / avg_duration if avg_duration > 0 else 0
                    }
                
                comparison["stage_comparisons"][stage_name] = stage_comparison
            
            # Overall process comparison
            successful_processes = []
            completion_times = []
            
            for i, snapshot in enumerate(process_snapshots):
                # Check if all stages completed
                stages = snapshot.get('stages', {})
                all_completed = all(stage.get('status') == 'completed' for stage in stages.values())
                
                if all_completed:
                    successful_processes.append(i)
                    
                    # Calculate total execution time if possible
                    stage_times = []
                    for stage in stages.values():
                        if stage.get('started_at') and stage.get('completed_at'):
                            started = pd.to_datetime(stage['started_at'])
                            completed = pd.to_datetime(stage['completed_at'])
                            stage_times.append((completed - started).total_seconds())
                    
                    if stage_times:
                        completion_times.append(sum(stage_times))
            
            comparison["overall_comparison"] = {
                "successful_processes": successful_processes,
                "success_rate": len(successful_processes) / len(process_snapshots) if process_snapshots else 0,
                "completion_times": completion_times,
                "average_completion_time": sum(completion_times) / len(completion_times) if completion_times else None,
                "fastest_run": min(completion_times) if completion_times else None,
                "slowest_run": max(completion_times) if completion_times else None
            }
            
            logger.info(f"Completed comparison of {len(process_snapshots)} process runs")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing process runs: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    @staticmethod
    def load_snapshot(filepath: str) -> Dict[str, Any]:
        """
        Load a process snapshot from a file
        
        Args:
            filepath: Path to the snapshot file
            
        Returns:
            Dictionary with the process snapshot
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
            return snapshot
        except Exception as e:
            logger.error(f"Error loading process snapshot: {str(e)}", exc_info=True)
            return {}