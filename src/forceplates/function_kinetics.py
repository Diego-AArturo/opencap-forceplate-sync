import os, sys
from pathlib import Path
from typing import List, Optional, Tuple

# 1) Añade las carpetas necesarias al “PYTHONPATH”
BASE_DIR         = Path(__file__).resolve().parent
OPENSIMAD_DIR    = BASE_DIR / 'UtilsDynamicSimulations' / 'OpenSimAD'
sys.path += [str(BASE_DIR), str(OPENSIMAD_DIR)]


from UtilsDynamicSimulations.OpenSimAD.utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from src.opensim.mainOpenSimAD  import run_tracking


def run_inverse_dynamics(
        *,
        session_id      : str,
        participant_id  : str,
        trial_name      : str,
        motion_type     : str = 'other',
        time_window     : Optional[Tuple[float,float]] = None,
        repetition      : Optional[int]   = None,
        treadmill_speed : float           = 0.0,
        contact_side    : str             = 'all',  # 'all', 'left', 'right'
        case            : str             = '0',
        solve_problem   : bool            = True,
        analyze_results : bool            = True,
        data_root       : str             = 'Data'
    ):
    """
    Ejecuta dinámica inversa con OpenSimAD para cualquier movimiento.

    • Si `motion_type` es uno de ['running', 'walking', 'squats',
      'sit_to_stand', 'drop_jump', 'jumping', ...] se usan los ajustes
      preconfigurados de OpenCap-processing.

    • Si `motion_type == 'other'` debes proporcionar `time_window=(t0,t1)`
      en segundos, relativo al tiempo del archivo .mot o de las coordenadas
      OpenSim exportadas por OpenCap.

    Parámetros mínimos:
        - session_id
        - participant_id (para que la carpeta se llame así)
        - trial_name
        - motion_type='other'
        - time_window=(t0,t1)
    """

    # 2) Carpeta donde guardarás todo.
    data_folder = Path(BASE_DIR, data_root, participant_id)
    data_folder.mkdir(parents=True, exist_ok=True)

    # 3) Validaciones básicas .............................................
    predef_types = [
        'running', 'walking', 'drop_jump', 'sit_to_stand',
        'squats', 'jumping', 'running_torque_driven', 'my_periodic_running'
    ]
    if motion_type not in predef_types + ['other']:
        raise ValueError(f"`motion_type` desconocido: {motion_type}")

    # Para tipos con segmentador automático
    if motion_type in ['squats', 'sit_to_stand'] and repetition is None:
        raise ValueError(
            f"`repetition` (int) es obligatorio cuando motion_type='{motion_type}'."
        )

    # Para movimiento libre ('other') exiges time_window
    # if motion_type == 'other':
    #     if time_window is None or len(time_window) != 2:
    #         raise ValueError(
    #             "`time_window=(t0,t1)` es obligatorio cuando motion_type='other'."
    #         )
    #     # OpenCap-processing espera None si NO lo defines.  Aquí ya lo tienes OK.
    # else:
    #     # Si hay time_window pero el tipo NO es 'other', se acepta (el usuario quizá
    #     # quiera limitar la simulación manualmente) -- no se hace nada.
    #     pass

    if contact_side not in ['all', 'left', 'right']:
        raise ValueError("contact_side debe ser 'all', 'left' o 'right'.")
    print('BASE_DIR: ', BASE_DIR)
    
    # 4) Construcción de settings (wrapper OpenCap) ........................
    settings = processInputsOpenSimAD(
        baseDir          = str(BASE_DIR),
        dataFolder       = str(data_folder),
        session_id       = session_id,
        trial_name       = trial_name,
        motion_type      = motion_type,
        time_window      = time_window,
        repetition       = repetition,
        treadmill_speed  = treadmill_speed,
        contact_side     = contact_side
    )

    # 5) Ejecutar la optimización (dinámica inversa vía tracking) ..........
    run_tracking(
        baseDir          = str(BASE_DIR),
        dataDir          = str(data_folder),
        subject          = session_id,
        settings         = settings,
        case             = case,
        solveProblem     = solve_problem,
        analyzeResults   = analyze_results
    )

    # 6) (Opcional) Post-processing / gráficas .............................
    plotResultsOpenSimAD(
        dataFolder       = str(data_folder),
        session_id       = session_id,
        trial_name       = trial_name,
        settings         = settings,
        cases            = [case]
    )

    print(f"✅  Simulación terminada. Resultados en:\n   {data_folder}")


# ----------------------------------------------------------------------------
# 🟢  EJEMPLO de uso para un movimiento NO predefinido
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_inverse_dynamics(
    session_id      = "da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38",
    participant_id  = "P1",
    trial_name      = "estocada_lateral_derecha_1",
    motion_type     = "other",   # porque no es 'walking', 'squats', etc.
    time_window     = None,      # toda la duración del trial
    case            = "0",
    solve_problem   = True,
    analyze_results = True
)
