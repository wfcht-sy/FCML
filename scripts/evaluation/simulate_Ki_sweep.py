#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import json
import glob
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import EVAL_RESULTS_DIR

# Controllers to evaluate
ALGOS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']

def calculate_rmse_from_log(log_filename):
    """Read actual simulation flight log to compute RMSE."""
    try:
        # time,p_x,v_x,pos_err_x,u_nom,u_comp,d_hat
        data = np.genfromtxt(log_filename, delimiter=',', skip_header=1)
        errors = data[:, 3]
        rmse = np.sqrt(np.mean(errors**2))
        return float(rmse)
    except Exception as e:
        print(f"Error reading {log_filename}: {e}")
        return float('inf')

def run_evaluation():
    log_dir = os.path.join(EVAL_RESULTS_DIR, 'Ki_sweep_logs')
    results = {}
    
    for algo in ALGOS:
        # Note: mapping 'FCML' to 'FCML' log files if they contain 'FCML'
        search_algo = 'FCML' if algo == 'FCML' else algo
        
        pattern = os.path.join(log_dir, f'eval_data_KiSweep_{search_algo}_Ki_*.csv')
        files = glob.glob(pattern)
        
        ki_values = []
        rmses = []
        
        for f in files:
            # Extract Ki from filename
            match = re.search(r'Ki_([0-9.]+)\.csv', f)
            if match:
                ki = float(match.group(1))
                rmse = calculate_rmse_from_log(f)
                ki_values.append(ki)
                rmses.append(rmse)
                
        # Sort by Ki
        if ki_values:
            sorted_indices = np.argsort(ki_values)
            ki_values = np.array(ki_values)[sorted_indices].tolist()
            rmses = np.array(rmses)[sorted_indices].tolist()
            
            min_idx = np.argmin(rmses)
            results[algo] = {
                'Ki_range': ki_values,
                'rmses': rmses,
                'best_Ki': ki_values[min_idx],
                'best_rmse': rmses[min_idx]
            }
            print(f"[{algo}] Evaluated {len(files)} logs. Optimal Ki: {ki_values[min_idx]:.2f}, Min RMSE: {rmses[min_idx]:.3f}")
        else:
            print(f"[{algo}] No simulation logs found.")
            
    return results

if __name__ == '__main__':
    print("Evaluating Ki parameter sweep from real simulation flight logs...")
    results = run_evaluation()

    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(EVAL_RESULTS_DIR, 'Ki_sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation finished. Results saved to: {json_path}")
