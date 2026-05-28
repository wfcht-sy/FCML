import torch
import pytorch_lightning as pl
import sys
import os

# Ensure paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modify0513', 'FCML')))

try:
    from scripts.offline.models import PhiNetwork, PhiNet, AdaptiveControllerModel
    from scripts.missions.online_mission_compare import BaseOffboardControl, BaselineController, get_controller
except ImportError:
    # Try alternate path if running from root
    from modify0513.FCML.scripts.offline.models import PhiNetwork, PhiNet, AdaptiveControllerModel
    from modify0513.FCML.scripts.missions.online_mission_compare import BaseOffboardControl, BaselineController, get_controller

def test_network_alignment():
    print("=== Testing Network Backbone Alignment ===")
    nf_model = PhiNetwork()
    fcml_model = PhiNet()
    
    x = torch.randn(2, 11)
    
    nf_out = nf_model(x)
    fcml_out = fcml_model(x)
    
    print("NF Output Shape:", nf_out.shape)
    print("FCML Output Shape:", fcml_out.shape)
    
    assert nf_out.shape == (2, 8), "NF model output shape mismatch!"
    assert fcml_out.shape == (2, 8), "FCML model output shape mismatch!"
    print("Forward Pass successful. Dimensions match.")
    print("NF Model:")
    print(nf_model)
    print("\nFCML Model:")
    print(fcml_model)
    print("=" * 40)

def test_lightning_checkpoint():
    print("\n=== Testing Lightning Checkpoint ===")
    model = AdaptiveControllerModel(controller_type='FCML', r_gain=9.9)
    # Use model's native save/load methods to test hyperparameter parsing
    import torch
    import lightning_fabric
    torch.serialization.add_safe_globals([lightning_fabric.utilities.data.AttributeDict])
    
    ckpt_path = "test_fcml.ckpt"
    ckpt = {
        "state_dict": model.state_dict(), 
        "hyper_parameters": model.hparams,
        "pytorch-lightning_version": pl.__version__
    }
    torch.save(ckpt, ckpt_path)
    
    loaded_model = AdaptiveControllerModel.load_from_checkpoint(ckpt_path, weights_only=False)
    print("Loaded Hyperparameters:", loaded_model.hparams)
    assert loaded_model.hparams.r_gain == 9.9, "Hyperparameter loading failed!"
    print("Lightning checkpoint save/load successful.")
    
    os.remove(ckpt_path)
    print("=" * 40)

def test_dry_run_controllers():
    print("\n=== Testing Offboard Controllers Dry-Run ===")
    import numpy as np
    state_cache = {
        'p': np.zeros(3),
        'v': np.zeros(3),
        'q': np.array([1.0, 0, 0, 0]),
        'pwm': np.array([0.70581]*4),
        'real_pwm': np.array([0.70581]*4)
    }
    import numpy as np
    
    for ctrl_type in ['Baseline', 'INDI', 'L1', 'Neural-Fly', 'FCML']:
        try:
            ctrl = get_controller(ctrl_type, state_cache, "0")
            roll, pitch, yaw, thrust = ctrl.compute(t=1.5)
            print(f"[{ctrl_type}] Output -> Roll: {roll:.2f}, Pitch: {pitch:.2f}, Thrust: {thrust:.2f}")
        except Exception as e:
            print(f"[{ctrl_type}] FAILED: {e}")
            raise e
    print("Dry-run computation successful for all controllers.")

if __name__ == "__main__":
    import numpy as np
    test_network_alignment()
    test_lightning_checkpoint()
    test_dry_run_controllers()
