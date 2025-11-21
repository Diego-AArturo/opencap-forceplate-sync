# Force Plate Integration with OpenCap

## Overview

This project provides tools for synchronizing force plate data with kinematic data from OpenCap. The main objective is to integrate temporal and spatial force plate measurements with OpenCap's marker-based motion capture system, enabling comprehensive biomechanical analysis.

## Main Objective

The primary goal of this system is to **synchronize OpenCap kinematic data with force plate measurements**, allowing researchers to combine:
- **Kinematic data** from OpenCap (marker positions, joint angles, etc.)
- **Kinetic data** from force plates (ground reaction forces, moments, center of pressure)

This synchronization enables comprehensive biomechanical analysis including inverse dynamics calculations in OpenSim.

## Quick Start

The main script to execute is **`scripts/batch_process_forceplates.py`**, which processes multiple participants and movements automatically.

## Step-by-Step Workflow

### Step 1: Preprocess Force Plate Files

Before processing, force plate files must be preprocessed using the Jupyter notebook:

**`notebooks/plataformas_fuerza.ipynb`**

This notebook:
- Processes raw force plate data files
- Standardizes column naming conventions
- Handles special cases (e.g., lateral sliding movements require leg label swapping)
- Prepares files for integration

**Important**: For lateral sliding movements (`deslizamiento_lateral`), the system automatically swaps right/left leg labels (`r_` ↔ `L_`, `l_` ↔ `R_`) to match the movement pattern.

### Step 2: Create Participant JSON Configuration File

After preprocessing, create a JSON file that relates force plate data to OpenCap video sessions. The movement names in the JSON **must exactly match** the trial names in OpenCap.

**File Structure** (example: `participantes.json`):

```json
[
    {
        "participant_id": "P1",
        "session_id": "opencap_session_id_here",
        "movements": [
            {
                "move": "movement_name",
                "link": "google_drive_url_or_empty"
            },
            {
                "move": "movement_name_2",
                "link": "google_drive_url_or_empty"
            }
        ]
    },
    {
        "participant_id": "P2",
        "session_id": "another_opencap_session_id",
        "movements": [
            {
                "move": "movement_name",
                "link": "google_drive_url_or_empty"
            }
        ]
    }
]
```

**Key Requirements**:
- `participant_id`: Unique identifier for the participant
- `session_id`: OpenCap session ID (found in OpenCap web interface)
- `move`: **Must exactly match** the trial name in OpenCap
- `link`: Google Drive URL for force plate data (can be empty if using local files)

### Step 3: Configure and Execute Batch Processing

1. Edit `scripts/batch_process_forceplates.py`:
   - Update `name_file` variable to match your JSON filename (without `.json` extension)
   - Configure movement classification sets (`unprocessed`, `bothlegs`) if needed

2. Run the script:
   ```bash
   python scripts/batch_process_forceplates.py
   ```

3. The script will:
   - Load participant data from the JSON file
   - Download or load force plate data
   - Download kinematic data from OpenCap (if not already present)
   - Synchronize temporal and spatial coordinates
   - Run inverse dynamics analysis in OpenSim
   - Generate synchronization plots

### Step 4: Access Synchronized Data

After processing, synchronized data can be found in:

```
Data/
└── {participant_id}/
    ├── MeasuredForces/
    │   └── {trial_name}/
    │       └── {trial_name}_syncd_forces.mot
    ├── OpenSimData/
    │   ├── InverseDynamics/
    │   │   └── {trial_name}/
    │   └── Kinematics/
    └── MarkerData/
```

## Integration Methods

The system provides two integration methods depending on the movement type and available information:

### Method 1: Single-Leg Integration
**File**: `src/forceplates/funtion_integrate_forceplates_legs.py`  
**Function**: `IntegrateForcepalte_legs()`

**Use when**:
- Only one foot contacts the force plate
- You know which foot contacts the plate first
- Movement is clearly unilateral (e.g., single-leg step-up, unilateral lunge)

**Parameters**:
- `legs`: Specify `'R'` for right leg or `'L'` for left leg

**Example**:
```python
IntegrateForcepalte_legs(
    session_id=session_id,
    trial_name="step_up_right_1",
    force_gdrive_url=force_url,
    participant_id="P1",
    legs='R'  # Specify the leg
)
```

### Method 2: Both-Leg Integration (Auto-Detection)
**File**: `src/forceplates/integrate_forceplates_both_legs.py`  
**Function**: `IntegrateForcepalte_vc0()`

**Use when**:
- Both feet contact force plates simultaneously
- You don't know which foot contacted first
- Movement is bilateral (e.g., squats, jumps, bilateral lunges)

**Features**:
- Automatically detects which leg initiated contact first
- Compares heel strike events from both legs
- Uses combined force signals for synchronization

**Example**:
```python
IntegrateForcepalte_vc0(
    session_id=session_id,
    trial_name="squat_90_1",
    force_gdrive_url=force_url,
    participant_id="P1"
    # No leg parameter needed - auto-detected
)
```

**Automatic Selection**: The batch processing script (`batch_process_forceplates.py`) automatically selects the appropriate method based on movement name patterns:
- Movements with `'derecha'` or `'derecho'` → Single-leg (right)
- Movements with `'izquierda'` or `'izquierdo'` → Single-leg (left)
- Movements in `bothlegs` set (e.g., `'sentadilla'`) → Both-leg integration

## Validation with Gold Standard

To validate synchronization results against a gold standard system (e.g., Motive motion capture with hardware-synchronized force plates), use the validation scripts in the `src/validation/` folder.

### Quick Validation Script
**File**: `src/validation/autocode4val.py`

This script processes multiple trials and compares OpenCap synchronized data with gold standard measurements.

### Comprehensive Validation
**File**: `src/validation/validation_sync.py`

The `SyncValidator` class provides comprehensive validation including:
- Temporal alignment comparison
- Force signal correlation analysis
- RMSE and MAE calculations
- Visualization plots
- Detailed validation reports

**Usage Example**:
```python
from src.validation.validation_sync import SyncValidator

validator = SyncValidator(
    gold_standard_mot_path="path/to/gold_forces.mot",
    gold_standard_trc_path="path/to/gold_markers.trc",
    opencap_syncd_mot_path="Data/P1/MeasuredForces/trial1/trial1_syncd_forces.mot",
    opencap_trc_path="Data/P1/MarkerData/trial1.trc",
    output_folder="validation_results"
)

metrics = validator.run_validation()
```

## Output Files

### Synchronized Force Data
- **Location**: `Data/{participant_id}/MeasuredForces/{trial_name}/{trial_name}_syncd_forces.mot`
- **Format**: OpenSim MOT file with synchronized force plate data
- **Content**: Ground reaction forces, moments, and center of pressure for both legs

### Inverse Dynamics Results
- **Location**: `Data/{participant_id}/OpenSimData/InverseDynamics/{trial_name}/{trial_name}.sto`
- **Format**: OpenSim STO file
- **Content**: Joint moments and powers calculated from synchronized data

### Synchronization Plots
- **Location**: `graficas/{participant_id}/{trial_name}_corte.png`
- **Content**: Visualization of heel position and synchronization point

### Lag Times
- **File**: `lag_times.json`
- **Content**: Temporal offset values calculated for each trial
- **Purpose**: Records synchronization parameters for reference

## Project Structure

```
ForcePlateIntegration/
├── scripts/
│   └── batch_process_forceplates.py      # Main batch processing script
├── src/
│   ├── forceplates/
│   │   ├── funtion_integrate_forceplates_legs.py    # Single-leg integration
│   │   └── integrate_forceplates_both_legs.py      # Both-leg integration
│   └── validation/
│       ├── validation_sync.py            # Comprehensive validation
│       └── autocode4val.py               # Quick validation script
├── notebooks/
│   └── plataformas_fuerza.ipynb          # Force plate preprocessing
├── Data/                                  # Output directory
└── README.md                              # This file
```

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies include:
- numpy
- pandas
- matplotlib
- scipy
- opensim
- requests

## Notes

- Movement names in the JSON configuration **must exactly match** OpenCap trial names
- Force plate files should be preprocessed before integration
- The system automatically handles coordinate system transformations
- Spatial calibration aligns force vectors with anatomical markers
- Temporal synchronization uses heel strike detection and cross-correlation

## Support

For issues or questions, please refer to the code documentation or contact the development team.

---

**Author**: ForcePlateIntegration Team
