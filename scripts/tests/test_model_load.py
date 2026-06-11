import torch
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from scripts.offline.models import PhiNetwork, PhiNet
ckpt_path = os.path.join(project_root, 'checkpoints', 'neural_fly_daiml_best.pth')
ckpt = torch.load(ckpt_path, map_location='cpu')

if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
else:
    state_dict = ckpt

print("Keys in state_dict:")
print(list(state_dict.keys())[:10])

# Since neural-fly used PhiNetwork originally:
model = PhiNetwork(input_dim=11, basis_dim=8)
try:
    model.load_state_dict(state_dict)
    print("PhiNetwork loaded successfully")
except Exception as e:
    print("PhiNetwork failed:", e)

model2 = PhiNet(input_dim=11, basis_dim=8)
try:
    model2.load_state_dict(state_dict)
    print("PhiNet loaded successfully")
except Exception as e:
    print("PhiNet failed:", e)
