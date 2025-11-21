"""
Batch processing script for force plate integration.

This script processes multiple participants and movements, integrating force plate
data with kinematic data. It automatically selects the appropriate integration method
based on movement type (single-leg vs both-leg movements) and generates error reports.

Author: ForcePlateIntegration Team
"""

from src.forceplates.funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from src.forceplates.integrate_forceplates_both_legs import IntegrateForcepalte_vc0

import json
import time


# Configuration
name_file = 'participantes4'
with open(f'{name_file}.json', 'r') as file:
    participantes = json.load(file)

# Tracking lists
fines = []  # Successful movements
fails = []  # Failed movements
fails_datail = []  # Detailed error information
es = []  # Unique error messages

# Movement classification
unprocessed = {
    'estocada_deslizamiento_lateral_derecho_1', 'estocada_deslizamiento_lateral_derecho_2',
    'estocada_deslizamiento_lateral_izquierdo_1', 'estocada_deslizamiento_lateral_izquierdo_2',
    'estocada_deslizamiento_posterior_derecho_1', 'estocada_deslizamiento_posterior_derecho_2',
    'estocada_deslizamiento_posterior_izquierdo_1', 'estocada_deslizamiento_posterior_izquierdo_2',
    'estocada_lateral_derecha_1', 'estocada_lateral_derecha_2',
    'estocada_lateral_izquierda_1', 'estocada_lateral_izquierda_2',
    'escalon_derecho_1', 'escalon_derecho_2', 'escalon_izquierdo_1', 'escalon_izquierdo_2',
    'estocada_derecha_1', 'estocada_derecha_2', 'estocada_izquierda_1', 'estocada_izquierda_2'
}

bothlegs = {
    'sentadilla_60_1', 'sentadilla_60_2', 'sentadilla_90_1', 'sentadilla_90_2',
}


def save_results(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data structure to save (dict or list)
        filename: Output filename (str)
    """
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_movement(usuario, session_id, movimiento, leg=None):
    """
    Process a movement and handle errors.
    
    Automatically selects the appropriate integration method:
    - Single-leg integration if leg is specified
    - Both-leg integration if leg is None
    
    Args:
        usuario: Participant ID (str)
        session_id: OpenCap session ID (str)
        movimiento: Movement dictionary with 'move' and 'link' keys (dict)
        leg: Leg identifier ('R' or 'L') for single-leg movements, None for both-leg (str, optional)
    """
    try:
        if leg:
            # Single-leg movement: use leg-specific integration
            IntegrateForcepalte_legs(
                session_id=session_id,
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id=usuario,
                legs=leg
            )
        else:
            # Both-leg movement: auto-detect initial contact leg
            IntegrateForcepalte_vc0(
                session_id=session_id,
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id=usuario
            )
        print(f"Success: {movimiento['move']}")
        fines.append((usuario, movimiento['move']))
    except Exception as e:
        fails.append((usuario, movimiento['move']))
        error_msg = f"{e}"
        fails_datail.append((usuario, movimiento['move'], error_msg))
        if error_msg not in es:
            es.append(error_msg)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

start_time = time.time()

# ============================================================================
# COMMENTED: Full batch processing for all participants
# ============================================================================
# Processes all participants and movements, automatically selecting integration
# method based on movement name (both-legs for squats, single-leg for others)
# for participante in participantes[:]:  # Process subset: participantes3 - 0:11 5
#     session_id = participante['session_id']
#     usuario = participante["participant_id"]
#     print(f'session_id {session_id}, participant {usuario}')
#     for movimiento in participante['movements']:
#         move_name = movimiento['move']
#
#         if move_name in unprocessed:
#             continue
#
#         if move_name in bothlegs:
#             process_movement(usuario, session_id, movimiento)
#         elif any(word in move_name for word in ['derecha', 'derecho']):
#             process_movement(usuario, session_id, movimiento, leg='R')
#         elif any(word in move_name for word in ['izquierda', 'izquierdo']):
#             process_movement(usuario, session_id, movimiento, leg='L')
#     print(f'Finished participant {usuario}')
#
# reporte = {
#     "resumen": {
#         "total_movimientos": len(fines) + len(fails),
#         "exitosos": len(fines),
#         "fallidos": len(fails),
#         "errores_unicos": len(set(es))
#     },
#     "movimientos_fallidos": fails_datail,
# }
# save_results(reporte, f"errores_{name_file}_todo.json")
#
# print('Total time:', (time.time() - start_time) / 60)
# print('Successful:', fines)
# print('Failed:', fails)
# print("Unique error types:", len(es))
# print('Errors\n', es)

# ============================================================================
# COMMENTED: Unit tests for specific movements
# ============================================================================
# Individual movement processing for testing/debugging specific cases
# process_movement('P22', 'c71d092b-93ba-4dfb-afc3-57fb46ef0736', {
#     "move": "estocada_deslizamiento_lateral_izquierdo_2",
#     "link": "https://drive.usercontent.google.com/u/0/uc?id=1ZGG6ZimaCZxS355mnO7r6QJ4jOXtJwFS&export=download"
# }, leg='L')
#
# process_movement('P9', '8cff7224-37bf-44d5-94d0-f7dfdea5bc36', {
#     "move": "estocada_deslizamiento_lateral_izquierdo_2",
#     "link": "https://drive.usercontent.google.com/u/0/uc?id=1OKm_wFbwnqtO7KIRHIcHrn6ifXo8gY0S&export=download"
# }, leg='L')
# error 23

# ============================================================================
# ACTIVE: Lateral lunge processing
# ============================================================================
# Currently processing lateral lunges (estocada_lateral) for all participants
for participante in participantes[:]:  # Process up to participant 11 (0-1 remaining: 2-4)
    session_id = participante['session_id']
    usuario = participante["participant_id"]
    print(f'session_id {session_id}, participant {usuario}')
    for movimiento in participante['movements']:
        if 'estocada_lateral_derecha' in movimiento['move'] or 'estocada_lateral_izquierda' in movimiento['move']:
            move_name = movimiento['move']
            process_movement(usuario, session_id, movimiento)
    print(f'Finished participant {usuario}')

# Generate and save report
reporte = {
    "resumen": {
        "total_movimientos": len(fines) + len(fails),
        "exitosos": len(fines),
        "fallidos": len(fails),
        "errores_unicos": len(set(es))
    },
    "movimientos_fallidos": fails_datail,
}

save_results(reporte, f"errores_{name_file}_estlat.json")

print('Total time:', (time.time() - start_time) / 60)
print('Successful:', fines)
print('Failed:', fails)
print("Unique error types:", len(es))
print('Errors\n', es)

# ============================================================================
# COMMENTED: Step-up movement processing
# ============================================================================
# Processes step-up movements (escalon) with automatic leg detection based on
# movement name (right/left leg identification)
# for participante in participantes[:]:  # Process up to participant 11 (0-1 remaining: 2-4)
#     session_id = participante['session_id']
#     usuario = participante["participant_id"]
#     print(f'session_id {session_id}, participant {usuario}')
#     for movimiento in participante['movements']:
#         if 'escalon' in movimiento['move']:
#             move_name = movimiento['move']
#             if any(word in move_name for word in ['derecha', 'derecho']):
#                 process_movement(usuario, session_id, movimiento, leg='R')
#             elif any(word in move_name for word in ['izquierda', 'izquierdo']):
#                 process_movement(usuario, session_id, movimiento, leg='L')
#     print(f'Finished participant {usuario}')
#
# reporte = {
#     "resumen": {
#         "total_movimientos": len(fines) + len(fails),
#         "exitosos": len(fines),
#         "fallidos": len(fails),
#         "errores_unicos": len(set(es))
#     },
#     "movimientos_fallidos": fails_datail,
# }
#
# save_results(reporte, f"errores_{name_file}_escalon.json")
# print('Total time:', (time.time() - start_time) / 60)
# print('Successful:', fines)
# print('Failed:', fails)
# print("Unique error types:", len(es))
# print('Errors\n', es)

