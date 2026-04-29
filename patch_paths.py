import os
import glob
import re

base_dir = r"f:\原始代码\最终版本代码整理\testmodel"

def replace_in_file(path, replacements):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
        
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {path}")

# 1. Update Bash Scripts
bash_scripts = glob.glob(os.path.join(base_dir, "scripts", "*.sh"))
bash_replacements = [
    ("/home/zzx/testmodel/Simulated_Data_Autocollect/set_wind.sh", "/home/zzx/testmodel/scripts/set_wind.sh"),
    ("python3 online_flight_figure81_compare.py", "python3 scripts/missions/online_flight_figure81_compare.py"),
    ("python3 online_mission_compare.py", "python3 scripts/missions/online_mission_compare.py"),
    ("python3 plot_comparison.py", "python3 scripts/evaluation/plot_comparison.py"),
    ("python3 plot_comparison_mission.py", "python3 scripts/evaluation/plot_comparison_mission.py"),
    ("python3 run_ablations.py", "python3 scripts/offline/run_ablations.py"),
    ("python3 plot_training_curve.py", "python3 scripts/evaluation/plot_training_curve.py"),
    ("python3 visualize_feature_clusters.py", "python3 scripts/evaluation/visualize_feature_clusters.py"),
    ("-f online_flight_figure81_compare.py", "-f scripts/missions/online_flight_figure81_compare.py"),
    ("-f online_mission_compare.py", "-f scripts/missions/online_mission_compare.py")
]
for script in bash_scripts:
    # Prepend cd to root at start
    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'cd "$(dirname "$0")/.."' not in content and script.endswith('run_evaluations.sh') or 'run_' in script:
        lines = content.split('\n')
        # Insert after #!/bin/bash
        if lines and lines[0].startswith('#!'):
            lines.insert(1, '\ncd "$(dirname "$0")/.." || exit\n')
            content = '\n'.join(lines)
            with open(script, 'w', encoding='utf-8') as f:
                f.write(content)
    
    replace_in_file(script, bash_replacements)

# 2. Update Python Imports (inject sys.path)
python_files = glob.glob(os.path.join(base_dir, "scripts", "**", "*.py"), recursive=True)
sys_path_code = """import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))\n"""

for py_file in python_files:
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    needs_sys_path = False
    
    if "from models import" in content or "import models" in content:
        content = content.replace("from models import", "from scripts.offline.models import")
        content = content.replace("import models", "from scripts.offline import models")
        needs_sys_path = True
    
    if needs_sys_path and "sys.path.append(" not in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if not line.startswith('#') and line.strip() != '':
                lines.insert(i, sys_path_code)
                break
        else:
            lines.insert(0, sys_path_code)
            
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Injected sys.path to {py_file}")

print("Done patching.")
