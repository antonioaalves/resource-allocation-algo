"""
Visualization module for allocation results
Provides functions for generating visual representations of allocation outcomes
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger('BagAllocationAlgo')

class ResultVisualizer:
    """Class for visualizing allocation results"""
    
    def __init__(self, output_dir: str = "data/output/visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory where visualization outputs will be saved
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"ResultVisualizer initialized with output directory: {output_dir}")
        
    def visualize_fill_bags_results(self, algorithm_result, output_filename: str = None) -> str:
        """
        Create visualizations for FillTheBags algorithm results
        
        Args:
            algorithm_result: FillTheBags algorithm result object
            output_filename: Optional filename for the output image
            
        Returns:
            Path to the saved visualization file
        """
        try:
            # Extract data from the algorithm result
            bag_allocations = algorithm_result.bag_allocations
            unused_balls = algorithm_result.unused_balls_ids
            
            # Prepare data for visualization
            bag_ids = []
            bag_colors = []
            capacities = []
            filled_percentages = []
            ball_counts = []
            
            for bag_id, allocation in bag_allocations.items():
                bag_ids.append(bag_id)
                bag_colors.append(allocation['color'])
                capacities.append(allocation['objective_capacity'])
                filled_percentages.append(
                    min(100, (allocation['filled_capacity'] / allocation['objective_capacity']) * 100)
                    if allocation['objective_capacity'] > 0 else 0
                )
                ball_counts.append(len(allocation['balls']))
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Fill The Bags Algorithm Results', fontsize=16)
            
            # Plot 1: Bag fill percentages
            ax1 = axes[0]
            bar_colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2', '#DECF3F', '#F15854']
            color_map = {color: bar_colors[i % len(bar_colors)] for i, color in enumerate(set(bag_colors))}
            
            bars = ax1.bar(
                bag_ids, 
                filled_percentages, 
                color=[color_map[color] for color in bag_colors]
            )
            
            # Add capacity target lines
            for i, (bag_id, capacity) in enumerate(zip(bag_ids, capacities)):
                ax1.plot([i-0.4, i+0.4], [100, 100], 'r--', alpha=0.7)
                
            ax1.set_title('Bag Fill Percentages')
            ax1.set_xlabel('Bag ID')
            ax1.set_ylabel('Fill Percentage (%)')
            ax1.set_ylim(0, 110)  # Leave room for the 100% line
            
            # Add value labels on bars
            for bar, percentage in zip(bars, filled_percentages):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 2,
                    f'{percentage:.1f}%',
                    ha='center', va='bottom', 
                    rotation=0, 
                    fontsize=8
                )
            
            # Add legend for bag colors
            legend_elements = [
                plt.Rectangle((0,0), 1, 1, color=color_map[color], label=color)
                for color in sorted(set(bag_colors))
            ]
            ax1.legend(handles=legend_elements, title="Bag Colors", loc='upper right')
            
            # Plot 2: Ball assignments per bag
            ax2 = axes[1]
            ax2.bar(
                bag_ids, 
                ball_counts,
                color=[color_map[color] for color in bag_colors]
            )
            
            # Add text annotation for unused balls
            unused_text = f'Unused Balls: {len(unused_balls)}'
            ax2.annotate(
                unused_text, 
                xy=(0.95, 0.95), 
                xycoords='axes fraction',
                fontsize=10, 
                backgroundcolor='white',
                ha='right', 
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
            
            ax2.set_title('Number of Balls Assigned to Each Bag')
            ax2.set_xlabel('Bag ID')
            ax2.set_ylabel('Number of Balls')
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save the figure
            if output_filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"fillbags_results_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Fill Bags visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating Fill Bags visualization: {str(e)}", exc_info=True)
            return "Visualization failed"
            
    def visualize_lp_results(self, algorithm_result, output_filename: str = None) -> str:
        """
        Create visualizations for LP algorithm results
        
        Args:
            algorithm_result: LP algorithm result object
            output_filename: Optional filename for the output image
            
        Returns:
            Path to the saved visualization file
        """
        try:
            # Extract data from the algorithm result
            results = algorithm_result.results
            status = algorithm_result.status
            
            if not results:
                logger.warning("No results available for LP visualization")
                return "No results available"
                
            # Prepare data for visualization
            schedule_data = []
            for (i, j, k), value in results['schedule'].items():
                if value > 0.5:  # Only include assigned values (account for floating point)
                    schedule_data.append({
                        'employee': i,
                        'production_line': j,
                        'time_period': k,
                        'value': 1
                    })
            
            if not schedule_data:
                logger.warning("No schedule assignments found for LP visualization")
                return "No assignments found"
                
            schedule_df = pd.DataFrame(schedule_data)
            
            # Extract understaffing and overstaffing data
            understaffing = []
            for (j, k), value in results['understaffing'].items():
                if value > 0:
                    understaffing.append({
                        'prod_line': j,  # Changed from 'production_line' to 'prod_line'
                        'time_period': k,
                        'value': value
                    })
                    
            overstaffing = []
            for (j, k), value in results['overstaffing'].items():
                if value > 0:
                    overstaffing.append({
                        'prod_line': j,  # Changed from 'production_line' to 'prod_line'
                        'time_period': k,
                        'value': value
                    })
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'LP Algorithm Results (Status: {status})', fontsize=16)
            
            # Plot 1: Assignment heatmap
            ax1 = axes[0]
            
            # Pivot the data for the heatmap
            pivot_df = schedule_df.pivot_table(
                index='employee', 
                columns='production_line', 
                values='value',
                aggfunc='sum',
                fill_value=0
            )
            
            # Create heatmap
            heatmap = ax1.imshow(pivot_df, cmap='Blues', aspect='auto')
            
            # Add labels
            ax1.set_title('Employee to Production Line Assignments')
            ax1.set_xlabel('Production Line ID')
            ax1.set_ylabel('Employee ID')
            
            # Add colorbar
            plt.colorbar(heatmap, ax=ax1, label='Assignment')
            
            # Set tick labels
            ax1.set_xticks(range(len(pivot_df.columns)))
            ax1.set_xticklabels(pivot_df.columns)
            ax1.set_yticks(range(len(pivot_df.index)))
            ax1.set_yticklabels(pivot_df.index)
            
            # Plot 2: Understaffing and overstaffing
            ax2 = axes[1]
            
            # Convert to DataFrames
            under_df = pd.DataFrame(understaffing)
            over_df = pd.DataFrame(overstaffing)
            
            # Check if DataFrames are empty
            if len(under_df) == 0 and len(over_df) == 0:
                ax2.text(0.5, 0.5, 'No understaffing or overstaffing', 
                        ha='center', va='center', fontsize=12)
            else:
                # Set up bar positions
                column_name = 'prod_line'  # Using the new column name
                
                # Get unique production lines
                prod_lines = []
                if not under_df.empty and column_name in under_df.columns:
                    prod_lines.extend(under_df[column_name].tolist())
                if not over_df.empty and column_name in over_df.columns:
                    prod_lines.extend(over_df[column_name].tolist())
                
                # Remove duplicates and sort
                prod_lines = sorted(set(prod_lines)) if prod_lines else []
                
                bar_width = 0.35
                x = np.arange(len(prod_lines))
                
                # Plot understaffing bars
                if not under_df.empty and column_name in under_df.columns:
                    under_values = [
                        under_df[under_df[column_name] == pl]['value'].sum()
                        if pl in under_df[column_name].values else 0
                        for pl in prod_lines
                    ]
                    ax2.bar(x - bar_width/2, under_values, bar_width, label='Understaffing', color='red', alpha=0.7)
                
                # Plot overstaffing bars
                if not over_df.empty and column_name in over_df.columns:
                    over_values = [
                        over_df[over_df[column_name] == pl]['value'].sum()
                        if pl in over_df[column_name].values else 0
                        for pl in prod_lines
                    ]
                    ax2.bar(x + bar_width/2, over_values, bar_width, label='Overstaffing', color='blue', alpha=0.7)
                
                # Add labels and legend
                ax2.set_title('Staffing Issues by Production Line')
                ax2.set_xlabel('Production Line ID')
                ax2.set_ylabel('Number of Staff')
                ax2.set_xticks(x)
                ax2.set_xticklabels(prod_lines)
                if prod_lines:  # Only add legend if we have data
                    ax2.legend()
            
            # Add objective value text
            ax2.text(
                0.95, 0.95, 
                f'Objective Value: {results["objective_value"]:.2f}', 
                ha='right', va='top',
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save the figure
            if output_filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"lp_results_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"LP visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating LP visualization: {str(e)}", exc_info=True)
        return "Visualization failed"
            

    
    def visualize_comparison(self, algorithm_results: Dict[str, Any], output_filename: str = None) -> str:
        """
        Create a comparison visualization between different algorithms
        
        Args:
            algorithm_results: Dictionary of algorithm results keyed by algorithm name
            output_filename: Optional filename for the output image
            
        Returns:
            Path to the saved visualization file
        """
        try:
            if not algorithm_results:
                logger.warning("No algorithm results provided for comparison")
                return "No results available"
                
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle('Algorithm Comparison', fontsize=16)
            
            # Extract metrics for comparison
            metrics = {}
            
            for algo_name, result in algorithm_results.items():
                if algo_name == 'fillbags':
                    # For FillTheBags, extract capacity utilization
                    bag_allocations = result.bag_allocations
                    total_capacity = sum(bag['objective_capacity'] for bag in bag_allocations.values())
                    filled_capacity = sum(bag['filled_capacity'] for bag in bag_allocations.values())
                    utilization = (filled_capacity / total_capacity * 100) if total_capacity > 0 else 0
                    
                    unused_balls = len(result.unused_balls_ids)
                    
                    metrics[algo_name] = {
                        'capacity_utilization': utilization,
                        'unused_resources': unused_balls
                    }
                    
                elif algo_name == 'lp':
                    # For LP, extract staffing metrics
                    if result.results:
                        objective_value = result.results['objective_value']
                        understaffing = sum(v for v in result.results['understaffing'].values())
                        overstaffing = sum(v for v in result.results['overstaffing'].values())
                        
                        metrics[algo_name] = {
                            'objective_value': objective_value,
                            'understaffing': understaffing,
                            'overstaffing': overstaffing,
                            'total_issues': understaffing + overstaffing
                        }
            
            # Plot comparison as bar chart - common metric: resource utilization
            if metrics:
                algo_names = list(metrics.keys())
                x = np.arange(len(algo_names))
                width = 0.35
                
                # For FillTheBags: capacity utilization
                # For LP: inverse of staffing issues (100 - normalized issues)
                performance_scores = []
                labels = []
                
                for algo in algo_names:
                    if algo == 'fillbags':
                        performance_scores.append(metrics[algo]['capacity_utilization'])
                        labels.append(f"Capacity Util.: {metrics[algo]['capacity_utilization']:.1f}%")
                    elif algo == 'lp':
                        # Normalize staffing issues to a percentage (lower is better)
                        # This is a simplified example - consider better normalization
                        total_issues = metrics[algo]['total_issues']
                        inverse_score = max(0, 100 - (total_issues * 10))  # Example scaling
                        performance_scores.append(inverse_score)
                        labels.append(f"Staffing Score: {inverse_score:.1f}%")
                
                # Create bar chart
                bars = ax.bar(x, performance_scores, width, label='Performance Score')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 1,
                        labels[i],
                        ha='center', va='bottom',
                        fontsize=10
                    )
                
                # Add labels
                ax.set_title('Algorithm Performance Comparison')
                ax.set_xlabel('Algorithm')
                ax.set_ylabel('Performance Score (%)')
                ax.set_xticks(x)
                ax.set_xticklabels(algo_names)
                ax.set_ylim(0, 110)  # Leave space for labels
                
                # Add additional metrics as text
                text_info = []
                for algo in algo_names:
                    if algo == 'fillbags':
                        text_info.append(
                            f"{algo} metrics:\n"
                            f"- Unused resources: {metrics[algo]['unused_resources']}\n"
                            f"- Capacity utilization: {metrics[algo]['capacity_utilization']:.1f}%"
                        )
                    elif algo == 'lp':
                        text_info.append(
                            f"{algo} metrics:\n"
                            f"- Objective value: {metrics[algo]['objective_value']:.2f}\n"
                            f"- Understaffing: {metrics[algo]['understaffing']:.1f}\n"
                            f"- Overstaffing: {metrics[algo]['overstaffing']:.1f}"
                        )
                
                # Add text box with metrics
                ax.text(
                    0.98, 0.02, 
                    '\n\n'.join(text_info), 
                    ha='right', va='bottom',
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
                )
            else:
                ax.text(0.5, 0.5, 'No comparable metrics available', 
                       ha='center', va='center', fontsize=12)
            
            # Save the figure
            if output_filename is None:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"algorithm_comparison_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Algorithm comparison visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating algorithm comparison: {str(e)}", exc_info=True)
            return "Visualization failed"
            
    def generate_html_report(self, algorithm_results: Dict[str, Any], process_summary: Dict[str, Any] = None) -> str:
        """
        Generate an HTML report with visualizations and results summary
        
        Args:
            algorithm_results: Dictionary of algorithm results keyed by algorithm name
            process_summary: Optional process summary dictionary
            
        Returns:
            Path to the saved HTML report
        """
        try:
            # Generate visualizations first
            viz_paths = {}
            
            for algo_name, result in algorithm_results.items():
                if algo_name == 'fillbags':
                    viz_paths[algo_name] = self.visualize_fill_bags_results(result)
                elif algo_name == 'lp':
                    viz_paths[algo_name] = self.visualize_lp_results(result)
            
            # Generate comparison if multiple algorithms
            if len(algorithm_results) > 1:
                viz_paths['comparison'] = self.visualize_comparison(algorithm_results)
            
            # Create HTML content
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Resource Allocation Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .viz-container {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f0f0f0; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .footer {{ margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; font-size: 0.8em; color: #777; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Resource Allocation Results</h1>
                    <p>Report generated on: {timestamp}</p>
                </div>
            """
            
            # Add process summary if available
            if process_summary:
                status_counts = process_summary.get('status_counts', {})
                completed = status_counts.get('completed', 0)
                failed = status_counts.get('failed', 0)
                
                html_content += f"""
                <div class="section">
                    <h2>Process Summary</h2>
                    <div class="summary-box">
                        <p><strong>Process ID:</strong> {process_summary.get('id', 'N/A')}</p>
                        <p><strong>Completed Stages:</strong> {completed}</p>
                        <p><strong>Failed Stages:</strong> {failed}</p>
                        <p><strong>Overall Progress:</strong> {process_summary.get('progress', 0) * 100:.1f}%</p>
                    </div>
                </div>
                """
            
            # Add algorithm results
            for algo_name, result in algorithm_results.items():
                html_content += f"""
                <div class="section">
                    <h2>{algo_name} Algorithm Results</h2>
                """
                
                # Add algorithm specific details
                if algo_name == 'fillbags':
                    bag_allocations = result.bag_allocations
                    total_capacity = sum(bag['objective_capacity'] for bag in bag_allocations.values())
                    filled_capacity = sum(bag['filled_capacity'] for bag in bag_allocations.values())
                    utilization = (filled_capacity / total_capacity * 100) if total_capacity > 0 else 0
                    unused_balls = len(result.unused_balls_ids)
                    
                    html_content += f"""
                    <div class="summary-box">
                        <p><strong>Capacity Utilization:</strong> {utilization:.1f}%</p>
                        <p><strong>Total Capacity:</strong> {total_capacity:.2f}</p>
                        <p><strong>Filled Capacity:</strong> {filled_capacity:.2f}</p>
                        <p><strong>Unused Balls:</strong> {unused_balls}</p>
                    </div>
                    
                    <h3>Bag Allocations</h3>
                    <table>
                        <tr>
                            <th>Bag ID</th>
                            <th>Color</th>
                            <th>Capacity</th>
                            <th>Filled</th>
                            <th>Utilization %</th>
                            <th>Assigned Balls</th>
                        </tr>
                    """
                    
                    for bag_id, bag in bag_allocations.items():
                        bag_util = (bag['filled_capacity'] / bag['objective_capacity'] * 100) if bag['objective_capacity'] > 0 else 0
                        ball_count = len(bag['balls'])
                        
                        html_content += f"""
                        <tr>
                            <td>{bag_id}</td>
                            <td>{bag['color']}</td>
                            <td>{bag['objective_capacity']:.2f}</td>
                            <td>{bag['filled_capacity']:.2f}</td>
                            <td>{bag_util:.1f}%</td>
                            <td>{ball_count}</td>
                        </tr>
                        """
                        
                    html_content += "</table>"
                    
                elif algo_name == 'lp':
                    if result.results:
                        objective_value = result.results['objective_value']
                        understaffing = sum(v for v in result.results['understaffing'].values())
                        overstaffing = sum(v for v in result.results['overstaffing'].values())
                        
                        html_content += f"""
                        <div class="summary-box">
                            <p><strong>Solver Status:</strong> {result.status}</p>
                            <p><strong>Objective Value:</strong> {objective_value:.2f}</p>
                            <p><strong>Total Understaffing:</strong> {understaffing:.1f}</p>
                            <p><strong>Total Overstaffing:</strong> {overstaffing:.1f}</p>
                        </div>
                        
                        <h3>Staffing Issues</h3>
                        <table>
                            <tr>
                                <th>Production Line</th>
                                <th>Time Period</th>
                                <th>Understaffing</th>
                                <th>Overstaffing</th>
                            </tr>
                        """
                        
                        # Combine understaffing and overstaffing data
                        staffing_issues = {}
                        
                        for (j, k), value in result.results['understaffing'].items():
                            if value > 0:
                                key = (j, k)
                                if key not in staffing_issues:
                                    staffing_issues[key] = {'under': 0, 'over': 0}
                                staffing_issues[key]['under'] = value
                                
                        for (j, k), value in result.results['overstaffing'].items():
                            if value > 0:
                                key = (j, k)
                                if key not in staffing_issues:
                                    staffing_issues[key] = {'under': 0, 'over': 0}
                                staffing_issues[key]['over'] = value
                        
                        for (j, k), values in sorted(staffing_issues.items()):
                            html_content += f"""
                            <tr>
                                <td>{j}</td>
                                <td>{k}</td>
                                <td>{values['under']:.1f}</td>
                                <td>{values['over']:.1f}</td>
                            </tr>
                            """
                            
                        html_content += "</table>"
                
                # Add visualization if available
                if algo_name in viz_paths and viz_paths[algo_name] != "Visualization failed":
                    # Convert to relative path for HTML
                    rel_path = os.path.relpath(viz_paths[algo_name], self.output_dir)
                    html_content += f"""
                    <div class="viz-container">
                        <h3>Visualization</h3>
                        <img src="{rel_path}" alt="{algo_name} visualization" style="max-width:100%; height:auto;">
                    </div>
                    """
                    
                html_content += "</div>"  # Close section
            
            # Add comparison visualization if available
            if 'comparison' in viz_paths and viz_paths['comparison'] != "Visualization failed":
                rel_path = os.path.relpath(viz_paths['comparison'], self.output_dir)
                html_content += f"""
                <div class="section">
                    <h2>Algorithm Comparison</h2>
                    <div class="viz-container">
                        <img src="{rel_path}" alt="Algorithm comparison" style="max-width:100%; height:auto;">
                    </div>
                </div>
                """
            
            # Add footer
            html_content += f"""
                <div class="footer">
                    <p>Resource Allocation System - Report generated on {timestamp}</p>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            timestamp_file = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"allocation_report_{timestamp_file}.html"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}", exc_info=True)
            return "Report generation failed"