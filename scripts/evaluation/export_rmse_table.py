import os
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import EVAL_RESULTS_DIR

RESULTS_DIR = EVAL_RESULTS_DIR
TABLE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'table_results')
os.makedirs(TABLE_DIR, exist_ok=True)

CONTROLLERS = ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']

WIND_CONDITIONS = {
    'online_test_nowind': '0', 
    'online_test_35wind': '4.2',
    'online_test_70wind': '8.5', 
    'online_test_100wind': '12.1',
    'online_test_70p20sint': '8.5+2.4sin(t)'
}

def load_data(ctrl, wind):
    file_path = os.path.join(RESULTS_DIR, f"eval_data_VirtualMission_{ctrl}_{wind}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def calculate_cross_track_errors(df_steady, ref_x, ref_y):
    pts = np.vstack((df_steady['p_x'], df_steady['p_y'])).T
    ref_pts = np.vstack((ref_x, ref_y)).T
    distances = np.min(np.linalg.norm(pts[:, np.newaxis, :] - ref_pts[np.newaxis, :, :], axis=2), axis=1)
    rmse = np.sqrt(np.mean(distances**2)) * 100.0
    mean_err = np.mean(distances) * 100.0
    return rmse, mean_err

theta_ref = np.linspace(0, 4 * np.pi, 2000)
ref_x = 4.0 * np.sin(theta_ref)
ref_y = 4.0 * np.sin(theta_ref) * np.cos(theta_ref)

rows = []
for ctrl in CONTROLLERS:
    row = {'Method': ctrl}
    for wind_key, wind_label in WIND_CONDITIONS.items():
        df = load_data(ctrl, wind_key)
        if df is not None and not df.empty:
            df_steady = df[(df['time'] >= 15.0) & (df['time'] <= 85.0)]
            if not df_steady.empty:
                rmse, mean_err = calculate_cross_track_errors(df_steady, ref_x, ref_y)
                row[f'{wind_label}_RMS'] = round(rmse, 1)
                row[f'{wind_label}_Mean'] = round(mean_err, 1)
            else:
                row[f'{wind_label}_RMS'] = np.nan
                row[f'{wind_label}_Mean'] = np.nan
        else:
            row[f'{wind_label}_RMS'] = np.nan
            row[f'{wind_label}_Mean'] = np.nan
    rows.append(row)

df_table = pd.DataFrame(rows)
cols = ['Method']
for wind_label in WIND_CONDITIONS.values():
    cols.append(f'{wind_label}_RMS')
    cols.append(f'{wind_label}_Mean')

df_table = df_table[cols]

output_file = os.path.join(TABLE_DIR, 'tracking_error_statistics.csv')
df_table.to_csv(output_file, index=False)
print(f"Table saved successfully to {output_file}")
print("\nGenerated Table Preview:")
print(df_table.to_string(index=False))
