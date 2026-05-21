import torch
import os
from scripts.offline.models import PhiNetwork, PhiNetworkFCML

ckpt_path = 'checkpoints/neural_fly_daiml_best.pth'
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

model2 = PhiNetworkFCML(input_dim=11, basis_dim=8)
try:
    model2.load_state_dict(state_dict)
    print("PhiNetworkFCML loaded successfully")
except Exception as e:
    print("PhiNetworkFCML failed:", e)
