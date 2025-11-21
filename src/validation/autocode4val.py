from src.forceplates.integrate_forceplates_both_legs import IntegrateForcepalte_vc0
import os
import json
import time


data = os.listdir('Data\P31T\MarkerData')
moves = [dat.split('.')[0] for dat in data if 'desl_lateral_izquierdo' in dat]

fines = []
fails = []
fails_datail = []

es = []

def process_movement(usuario, session_id, movimiento):
    """Procesa un movimiento y maneja errores."""
    try:
        IntegrateForcepalte_vc0(
            session_id=session_id,
            trial_name=movimiento,
            force_gdrive_url='',
            participant_id=usuario,
            use_local_forces=True
        )
        print(f"funcionó: {movimiento}")
        fines.append((usuario, movimiento))
    except Exception as e:
        fails.append((usuario, movimiento))
        error_msg = f"{e}"
        fails_datail.append((usuario, movimiento, error_msg))
        if error_msg not in es:
            es.append(error_msg)

iniciar = time.time()
session_id = "session_id" 
usuario = "P31T"
print(f'sesion_id {session_id}, usuario {usuario}')
for movimiento in moves:
    process_movement(usuario, session_id, movimiento)
print(f'Termino participante {usuario}')

reporte = {
    "resumen": {
        "total_movimientos": len(fines) + len(fails),
        "exitosos": len(fines),
        "fallidos": len(fails),
        "errores_unicos": len(set(es))
    },
    
    "movimientos_fallidos": fails_datail,

    
}


print('tiempo total:', (time.time()-iniciar)/60)
print('correctos:', fines)
print('errores:', fails)
print("type_error:", len(es))
print('errors\n', es)
