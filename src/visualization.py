import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Any
import json

def plot_training_evolution(csv_path: str, save_dir: str):
    """Plot training metrics evolution"""
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # CVaR quantile evolution
    ax1.plot(df['epoch'], df['eta'], 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('η (CVaR Quantile)')
    ax1.set_title('CVaR Quantile Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Average loss evolution
    ax2.plot(df['epoch'], df['avg_loss'], 'r-', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Average Loss')
    ax2.set_title('Loss Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Lambda weights evolution
    lam_cols = [c for c in df.columns if c.startswith('lam')]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, col in enumerate(lam_cols):
        ax3.plot(df['epoch'], df[col], color=colors[i % len(colors)], 
                linewidth=2, marker='o', label=f'λ_{i}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Adaptive Weights Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Steps per epoch
    if 'steps' in df.columns:
        ax4.bar(df['epoch'], df['steps'], alpha=0.7, color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Successful Steps')
        ax4.set_title('Training Steps per Epoch')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_cost_breakdown(results: Dict[str, Any], save_path: str):
    """Plot cost component breakdown by method"""
    methods = list(results.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        return
    
    # Assume 4 cost components: jerk_rms, acc_rms, smooth, lane_dev
    cost_names = ['RMS Jerk', 'RMS Accel', 'Smoothness', 'Lane Deviation']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, cost_name in enumerate(cost_names):
        method_values = []
        method_names = []
        
        for method in methods:
            if f'{method}_cost_{i}' in results:
                method_values.append(results[f'{method}_cost_{i}'])
                method_names.append(method.title())
        
        if method_values:
            axes[i].bar(method_names, method_values, alpha=0.7)
            axes[i].set_ylabel('Cost Value')
            axes[i].set_title(cost_name)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_scenario_performance(scenario_results: Dict[str, Dict], save_path: str):
    """Plot performance breakdown by scenario type"""
    scenarios = list(scenario_results.keys())
    if not scenarios:
        return
        
    methods = list(scenario_results[scenarios[0]].keys())
    n_scenarios = len(scenarios)
    n_methods = len(methods)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ADE by scenario
    x = np.arange(n_scenarios)
    width = 0.25
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, method in enumerate(methods):
        ade_values = [scenario_results[scenario][method]['ade'] for scenario in scenarios]
        ax1.bar(x + i*width, ade_values, width, label=method.title(), 
               color=colors[i % len(colors)], alpha=0.7)
    
    ax1.set_xlabel('Scenario Type')
    ax1.set_ylabel('ADE (m)')
    ax1.set_title('ADE by Scenario Type')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FDE by scenario
    for i, method in enumerate(methods):
        fde_values = [scenario_results[scenario][method]['fde'] for scenario in scenarios]
        ax2.bar(x + i*width, fde_values, width, label=method.title(),
               color=colors[i % len(colors)], alpha=0.7)
    
    ax2.set_xlabel('Scenario Type')
    ax2.set_ylabel('FDE (m)')
    ax2.set_title('FDE by Scenario Type')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_training_report(config_path: str, results_dir: str, output_path: str):
    """Generate comprehensive training report"""
    import yaml
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load training log
    csv_path = os.path.join(results_dir, 'train_log.csv')
    train_data = None
    if os.path.exists(csv_path):
        train_data = pd.read_csv(csv_path)
    
    # Load evaluation results
    eval_path = os.path.join(results_dir, 'eval_results.json')
    eval_data = None
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Meta-CVaR Training Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-name {{ font-weight: bold; }}
            .metric-value {{ color: #2E86AB; font-size: 1.2em; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Two-Stage CVaR Meta-Learning Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Configuration Summary</h2>
            <div class="metric">
                <span class="metric-name">CVaR α:</span>
                <span class="metric-value">{cfg['training']['cvar_alpha']}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Learning Rate:</span>
                <span class="metric-value">{cfg['training']['learning_rate']}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Cost Dimensions:</span>
                <span class="metric-value">{cfg['model']['cost_dim']}</span>
            </div>
        </div>
    """
    
    if train_data is not None and len(train_data) > 0:
        final_row = train_data.iloc[-1]
        html_content += f"""
        <div class="section">
            <h2>Training Results</h2>
            <div class="metric">
                <span class="metric-name">Final η:</span>
                <span class="metric-value">{final_row['eta']:.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Final Loss:</span>
                <span class="metric-value">{final_row['avg_loss']:.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Epochs:</span>
                <span class="metric-value">{final_row['epoch']}</span>
            </div>
            
            <h3>Final Lambda Weights</h3>
            <table>
                <tr><th>Component</th><th>Weight</th><th>Interpretation</th></tr>
        """
        
        interpretations = ['RMS Jerk', 'RMS Acceleration', 'Smoothness', 'Lane Deviation']
        lam_cols = [c for c in train_data.columns if c.startswith('lam')]
        for i, col in enumerate(lam_cols):
            interp = interpretations[i] if i < len(interpretations) else f'Component {i}'
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{final_row[col]:.4f}</td>
                    <td>{interp}</td>
                </tr>
            """
        html_content += "</table>"
    
    if eval_data is not None:
        html_content += f"""
        <div class="section">
            <h2>Evaluation Results</h2>
            <table>
                <tr><th>Method</th><th>ADE (m)</th><th>FDE (m)</th></tr>
        """
        
        methods = ['uniform', 'comfort', 'adaptive']
        for method in methods:
            if f'{method}_ade' in eval_data:
                html_content += f"""
                <tr>
                    <td>{method.title()}</td>
                    <td>{eval_data[f'{method}_ade']:.3f}</td>
                    <td>{eval_data[f'{method}_fde']:.3f}</td>
                </tr>
                """
        html_content += f"""
            </table>
            <p>Evaluated on {eval_data.get('n_samples', 0)} samples</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)