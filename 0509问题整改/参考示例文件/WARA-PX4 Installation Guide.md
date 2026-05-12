# WARA: Wavelet-based Realtime Recovery against Acoustic Injection Attacks on UAV

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Platform](https://img.shields.io/badge/Platform-Ubuntu_20.04-orange)

WARA is an effective and low-overhead acoustic attack mitigation technique.
It recovers benign sensor signals from compromised MEMS gyroscopes and accelerometers.

This repository holds the dataset, scripts, and the documentation to reproduce the figures and experiment results in our paper.
For the WARA-enhanced PX4 Autopilot, please refer to `Firmware/README.md`.

Here is the file structure:
```
WARA-PX4
├── data  <--- Datasets
├── Firmware  <--- Codebase & Setup Documentation
├── logger_topics  <--- Log Configuration
│   Topics to be recorded by PX4-Autopilot.
│   For simulation, copy to build/px4_sitl_wavelet/etc/logging/logger_topics.txt
│   See https://docs.px4.io/v1.13/en/dev_log/logging.html#sd-card-configuration
│
├── missions  <--- Mission Plan for Simulation
│   For evaluation, load "Complex Manuevers.plan" to QGroundControl
│
├── parameters  <--- Experiment parameters used by the batch_experiment.py
├── scripts  <--- Python scripts for Data Processing, Offline Evaluation, and Plot Figures.
├── unrocker  <--- Host UnRocker models for the offline evaluation.
├── utils  <--- Various helper functions for the script
└── batch_experiments.py <-- Automated Log Collection to reproduce SITL experiments (Experimental!)
```

## Setup

We developed and tested WARA on **Ubuntu 20.04 LTS** and **Python 3.10**.

For developer and end-user:
1. **Create a virtual environment** to manage python dependencies:
   - Create a new Python virtual environment: `python3.10 -m venv .env`.
   - Activate the virtual environment: `source .env/bin/activate`.
2. Read `Firmware/README.md` to **install and build** WARA-PX4.

For user that also have interest in reproduce the experiment results in the `Artifact Evaluation Guide`:
1. Complete _all_ previous steps to install WARA-PX4.
2. Install python dependencies for scripts through:
    ```bash
    pip install -r requirements.txt
    ```
3. Read `Firmware/README-UnRocker.md` to **install and build** UnRocker-PX4.
4. **Unzip compressed dataset** to corresponding folders before plotting these figures:
   1. Figs. 3 & 7: `./data/HITL_ONLINE.zip.00*` -> `./data/HITL_ONLINE/`
   2. Fig. 6: `./data/SITL_DATA.zip.00*` -> `./data/SITL_DATA/`
   3. Figs. 16-19: `./data/SITL_WAVEFORM_EFFECTS.zip.00*` -> `./data/SITL_WAVEFORM_EFFECTS/`

## Artifact Evaluation Guide

### Primary Functionality Evaluation

#### Running WARA without Attack

_[1 human-minute + 10 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run this command to enable WARA in normal flight without attack.
    ```bash
    python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_NoAttack.yaml"
    ```
  The script will automatically start the autopilot twice:
  - The first launch for setting flight parameters (about 15-60 s). 
  - The demonstration will begin at the second launch. The simulator will show the drone's flight status while the `Terminal` console display messages from the autopilot.
  - For Gazebo simulator, following the drone with:
    - On the left panel, expand `Models` by clicking the triangle button.
    - Right click `iris` and select `follow`, the camera will then follow the drone.

  The whole demonstration can be stopped by closing all popped console windows.

- Result: The drone with WARA-PX4 flies normally without interruption in the flight mission.

#### Online Recovery against Simulated Gyro Attack

_[2 human-minutes + 15 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run this command to verify WARA's recovery performance against gyroscope attack.
    ```bash
    python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml"
    ```
  Override the detector parameter file setting to disable WARA's recovery:
    ```bash
    python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml" --override param.detector="parameters/solution/wara_sitl/disable_detector.yaml"
    ```
  Each execution will automatically start the autopilot twice to apply new flight parameters.
  The whole demonstration can be stopped by closing all popped console windows.
  To test other attack settings, override the `attack.param_file` configuration:
  ```bash
  # Add these configuration after the --override flag
  # Gyro 4.0 rad/s 20 Hz
  attack.param_file="parameters/attack/gyro_attack/GyroAttack_4.0_20.0Hz.yaml"
  # Gyro 4.0 rad/s 30 Hz
  attack.param_file="parameters/attack/gyro_attack/GyroAttack_4.0_30.0Hz.yaml"
  # Gyro 4.0 rad/s 40 Hz
  attack.param_file="parameters/attack/gyro_attack/GyroAttack_4.0_40.0Hz.yaml"
  ```

- Result: Despite the gyro attack, the drone with WARA enabled can avoid imminent crash and continue the flight with moderate oscillation.
          In contrast, the drone with WARA disabled will crash immediately.

#### Online Recovery against Side-Swing and Switch Attacks

_[4 human-minutes + 30 compute-minutes]_

Verify the recovery performance of WARA-PX4 against advanced gyroscope attacks (Side-Swing and Switch attacks) in SITL simulation flights.

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Use these commands to verify WARA's recovery performance against Side-Swing and Switch attacks.
    ```bash
      # WARA against Side-Swing Attack
      python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml" --override attack.param_file="parameters/attack/gyro_attack_modulated/Side-Swing_Attack.yaml"
      # WARA against Switch Attack
      python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml" --override attack.param_file="parameters/attack/gyro_attack_modulated/Switch_Attack.yaml"
    ```
  Similar to the previous case, use these commands to disable WARA and see what will happen:
    ```bash
      # Side-Swing Attack without WARA
      python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml" --override param.detector="parameters/solution/wara_sitl/disable_detector.yaml" attack.param_file="parameters/attack/gyro_attack_modulated/Side-Swing_Attack.yaml"
      # Switch Attack without WARA
      python batch_experiments.py "./Firmware/WARA-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_WARA_GyroAttack.yaml" --override param.detector="parameters/solution/wara_sitl/disable_detector.yaml" attack.param_file="parameters/attack/gyro_attack_modulated/Switch_Attack.yaml"
    ```

- Result: Despite these attacks, the drone with WARA enabled can avoid imminent crash and continue the flight with moderate oscillation.
          In contrast, the drone with WARA disabled will crash immediately.

#### Evaluate UnRocker with Attack

_[1 human-minute + 5 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run this command to enable UnRocker with gyroscope attack.
    ```bash
    python batch_experiments.py "./Firmware/UnRocker-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_UnRocker_GyroAttack.yaml"
    ```

- Result: The drone with UnRocker crashed after the attack is begun.

#### Evaluate Heuristic Filters with Attack _[3 human-minutes + 15 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following commands separately to enable various heuristic filters with gyroscope attack.
    ```bash
    # Butterworth Filter
    python batch_experiments.py "./Firmware/UnRocker-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_Butterworth_GyroAttack.yaml"
    # Savitzky-Golay Filter
    python batch_experiments.py "./Firmware/UnRocker-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_SavGol_GyroAttack.yaml"
    # Wiener Filter
    python batch_experiments.py "./Firmware/UnRocker-PX4/" "./missions/Complex Maneuvers.plan" --config "./parameters/experiments/SITL_Wiener_GyroAttack.yaml"
    ```

- Result: The drone with Butterworth and Wiener filters crashed immediately after the attack.
  Savitzky-Golay filter can maintain the flight attitude occasionally, but its recovery duration is significantly shorter than WARA.

### Result Reproduction

#### Touch-based Acoustic Attack vs. the Physical Shielding on CUAV V5+ (Fig. 2)

_[1 human-minute + 1 compute-minute]_

Demonstrate our touch-based acoustic attack can penetrate the physical shielding on CUAV V5+
and distort the drone motion signal.

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.2 Real-world attack on CUAV V5+.py"
    ```

[//]: # (- Result: As shown in Fig. 2, despite physical shielding, the waveform and frequency of the error)
[//]: # (          introduced by acoustic attacks are comparable to those of the benign signal recorded by)
[//]: # (          the ICM-20689 sensor, indicating that the acoustic attack can distort the drone motion signal.)

#### Impact of Computational Delay on Online Gyroscope Recovery (Fig. 3)

_[1 human-minute + 1 compute-minute]_
Demonstrate the impact of computational delay on the online recovery performance with the UnRocker and WARA in HITL simulation.

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
  Then unzip volumes of `HITL_ONLINE` (located in `./data`) to folder `./data/HITL_ONLINE`.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.3 Effect of computational delay in gyro recovery.py"
    ```

[//]: # (- Result:  Fig. 3 presents the online recovery results in a gyroscope attack, with an amplitude $G_i = 1.0\ rad/s$ and)
[//]: # (induced frequency $f_i = 20\ Hz$. The measurement recovered by UnRocker quickly deviated from the actual state,)
[//]: # (resulting in an immediate crash. Even without any attack &#40;$G_i=0$&#41;, the drone still loses control quickly, which)
[//]: # (means the data processing delay significantly degrades UnRocker’s online recovery performance.)
[//]: # (In contrast, WARA’s recovered measurements are more consistent with the actual state.)

#### Offline Comparison with UnRocker on Remote Acoustic Attack Dataset (Fig. 4&5)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.4&5 Remote Attack Recovery (Real-World).py"
    ```

[//]: # (- Result: Fig. 4 illustrates the offline recovery results under the)
[//]: # (remote gyroscope attacks. Fig. 5 shows the corresponding noise amplitude spectrum.)
[//]: # (The reduced fluctuations in the sensor data &#40;Fig. 4&#41; and the diminished peaks)
[//]: # (of attacked signals &#40;Fig. 5&#41; confirm that both Wara and UnRocker effectively)
[//]: # (suppress acoustic injection attacks. However, UnRocker’s recovered)
[//]: # (measurements deviate from the drone’s actual state. This deviation)
[//]: # (becomes significant with large amplitude sensor signals. For in-)
[//]: # (stance, in Fig. 4, UnRocker returns near-zero pitch rates even when)
[//]: # (the actual state exceeds 1.0 rad/s. The distortion peaks in Fig. 5 also)
[//]: # (confirm that UnRocker’s recovery process distorts benign signals.)
[//]: # (In contrast, WARA’s orthogonal wavelet packet transform effec-)
[//]: # (tively preserves drone motion signals while efficiently removing)
[//]: # (the induced signals. This capability significantly enhances WARA’s)
[//]: # (resilience against acoustic attacks.)

#### Offline Comparison with UnRocker on Touch-based Acoustic Attack Dataset (Figs. 26&27)

_[2 human-minutes + 4 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.26 Recovered gyroscope measurements (Touch-based real-world attack).py"
      python "scripts/plot_figure/Fig.27 Noise amplitude spectra of gyroscope recovery (Touch-based real-world attack).py"
    ```

#### Online Comparison with Heuristic Filters in SITL Simulation (Fig. 6)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**. 
  Then unzip volumes of `SITL_DATA` (located in `./data`) to folder `./data/SITL_DATA`.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.6 Recovered gyro measurement - Heuristic SITL Online.py"
    ```

[//]: # (- Result: )

#### Online Comparison with VIMU in HITL Simulation (Fig. 8)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
  Then unzip volumes of `HITL_ONLINE` (located in `./data`) to folder `./data/HITL_ONLINE`.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.8 Flight Attitude in Gyro Recovery - VIMU HITL Online.py"
    ```

[//]: # (- Result: )

#### Online Comparison with SpecGuard in SITL Simulation (Fig. 9)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.9 Flight Altitude in Gyro Recovery - SpecGuard HITL Online.py"
    ```

[//]: # (- Result: )

#### Impact of Attack Waveform (Figs. 18-21)

_[2 human-minutes + 4 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
  Then unzip volumes of `SITL_WAVEFORM_EFFECTS` (located in `./data`) to folder `./data/SITL_WAVEFORM_EFFECTS`.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.18&19 Effect of Waveform in Recovery.py"
      python "scripts/plot_figure/Fig.20&21 Comparison Against Modulated Attack.py"
    ```

#### Consistency in Simulated and Remote Attack (Fig. 11)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.11 Comparison of the simulated and remote attack.py"
    ```

#### Impact of WPT Parameters to Performance (Fig. 16)

_[1 human-minute + 1 compute-minute]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.16 WPT Nodes Impact on Runtime Overheads.py"
    ```

#### Impact of External Disturbance (Fig. 22-24)

_[2 human-minutes + 4 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run the following script to produce the figure in `data/figures`:
    ```bash
      python "scripts/plot_figure/Fig.22&23 Parameter Selection in Wind Disturbance.py"
      python "scripts/plot_figure/Fig.24 TTD at Different Wind Settings.py"
    ```

#### Supplementary Offline Comparison Results (Fig. 28-33)

_[6 human-minutes + 12 compute-minutes]_

- Preparation: Open a shell in the provided repo folder and activate the virtual environment described in the **Setup**.
- Execution: Run plotting scripts with different options to produce these figures in `data/figures`:
    ```bash
      # Fig. 28
      python "scripts/plot_figure/Fig.XX plot_lineplot_grid.py" --plot_data_names Attack WARA UnRocker --topic_name sensor_accel --attack_amplitude 20.0 --time_slice 6987 7007 --subplot_height 1.6 --layout_rect 0 0.06 1.0 1.0
      # Fig. 30
      python "scripts/plot_figure/Fig.XX plot_lineplot_grid.py" --plot_data_names Attack Butterworth Sav-Gol WARA Wiener Groundtruth --time_slice 6987 7007 --subplot_height 2.0 --layout_rect 0 0.06 1.0 1.0
      # Fig. 32
      python "scripts/plot_figure/Fig.XX plot_lineplot_grid.py" --plot_data_names Attack Butterworth Sav-Gol WARA Wiener Groundtruth --topic_name sensor_accel --attack_amplitude 20.0 --time_slice 6987 7007 --subplot_height 2.0 --layout_rect 0 0.06 1.0 1.0
    ```
    Replace the script with `scripts/plot_figure/Fig.XX plot_fftplot_grid.py` to plot their corresponding amplitude spectra:
    ```bash
      # Fig. 29
      python "scripts/plot_figure/Fig.XX plot_fftplot_grid.py" --plot_data_names Attack WARA UnRocker --topic_name sensor_accel --attack_amplitude 20.0 --time_slice 6987 7007 --subplot_height 1.6 --layout_rect 0 0.06 1.0 1.0
      # Fig. 31
      python "scripts/plot_figure/Fig.XX plot_fftplot_grid.py" --plot_data_names Attack Butterworth Sav-Gol WARA Wiener Groundtruth --time_slice 6987 7007 --subplot_height 2.0 --layout_rect 0 0.06 1.0 1.0
      # Fig. 33
      python "scripts/plot_figure/Fig.XX plot_fftplot_grid.py" --plot_data_names Attack Butterworth Sav-Gol WARA Wiener Groundtruth --topic_name sensor_accel --attack_amplitude 20.0 --time_slice 6987 7007 --subplot_height 2.0 --layout_rect 0 0.06 1.0 1.0
    ```
