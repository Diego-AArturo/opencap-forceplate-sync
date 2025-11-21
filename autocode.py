from funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from funtion_clean_forceplates0 import IntegrateForcepalte_vc0


import json
import time
# from data_participantes import participantes


name_file = 'participantes4'
with open(f'{name_file}.json', 'r') as file:
    participantes = json.load(file)

fines = []
fails = []
fails_datail = []

es = []


unprocessed = {
            'estocada_deslizamiento_lateral_derecho_1','estocada_deslizamiento_lateral_derecho_2','estocada_deslizamiento_lateral_izquierdo_1', 'estocada_deslizamiento_lateral_izquierdo_2',
            'estocada_deslizamiento_posterior_derecho_1','estocada_deslizamiento_posterior_derecho_2','estocada_deslizamiento_posterior_izquierdo_1', 'estocada_deslizamiento_posterior_izquierdo_2',
            #'sentadilla_60_1','sentadilla_60_2','sentadilla_90_1','sentadilla_90_2',
            'estocada_lateral_derecha_1','estocada_lateral_derecha_2','estocada_lateral_izquierda_1','estocada_lateral_izquierda_2'
            'escalon_derecho_1','escalon_derecho_2','escalon_izquierdo_1','escalon_izquierdo_2'
            'estocada_derecha_1','estocada_derecha_2','estocada_izquierda_1','estocada_izquierda_2'
            }
bothlegs = {
    # 'estocada_lateral_derecha_1','estocada_lateral_derecha_2','estocada_lateral_izquierda_1','estocada_lateral_izquierda_2'
     'sentadilla_60_1','sentadilla_60_2','sentadilla_90_1','sentadilla_90_2',
    
    }

def guardar_resultados(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_movement(usuario, session_id, movimiento, leg=None):
    """Procesa un movimiento y maneja errores."""
    try:
        if leg:
            IntegrateForcepalte_legs(
                session_id=session_id,
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id = usuario,
                legs=leg
            )
        else:
            IntegrateForcepalte_vc0(
                session_id=session_id,
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id = usuario
            )
        print(f"funcionó: {movimiento['move']}")
        fines.append((usuario,movimiento['move']))
    except Exception as e:
        fails.append((usuario,movimiento['move']))
        error_msg = f"{e}"
        fails_datail.append((usuario,movimiento['move'], error_msg))
        if error_msg not in es:
            es.append(error_msg)

iniciar = time.time()
# for participante in participantes[:]: #participantes3 - 0:11 5
#     session_id = participante['session_id']
#     usuario = participante["participant_id"]
#     print(f'sesion_id {session_id}, usuario {usuario}')
#     for movimiento in participante['movements']:
#         move_name = movimiento['move']

#         if move_name in unprocessed:
#             continue

#         if move_name in bothlegs:
#             process_movement(usuario,session_id, movimiento)
#         elif any(word in move_name for word in ['derecha', 'derecho']):
#             process_movement(usuario,session_id, movimiento, leg='R')
#         elif any(word in move_name for word in ['izquierda', 'izquierdo']):
#             process_movement(usuario,session_id, movimiento, leg='L')
#     print(f'Termino participante {usuario}')

# reporte = {
#     "resumen": {
#         "total_movimientos": len(fines) + len(fails),
#         "exitosos": len(fines),
#         "fallidos": len(fails),
#         "errores_unicos": len(set(es))
#     },
    
#     "movimientos_fallidos": fails_datail,

    
# }
# guardar_resultados(reporte, f"errores_{name_file}_todo.json")

# print('tiempo total:', (time.time()-iniciar)/60)
# print('correctos:', fines)
# print('errores:', fails)
# print("type_error:", len(es))
# print('errors\n', es)

#=============prueba unitaria==================

# process_movement('P22','c71d092b-93ba-4dfb-afc3-57fb46ef0736',{
#                 "move": "estocada_deslizamiento_lateral_izquierdo_2",
#                 "link": "https://drive.usercontent.google.com/u/0/uc?id=1ZGG6ZimaCZxS355mnO7r6QJ4jOXtJwFS&export=download"
#             }, leg='L')

# process_movement('P9','8cff7224-37bf-44d5-94d0-f7dfdea5bc36',{
            #     "move": "estocada_deslizamiento_lateral_izquierdo_2",
            #     "link": "https://drive.usercontent.google.com/u/0/uc?id=1OKm_wFbwnqtO7KIRHIcHrn6ifXo8gY0S&export=download"
            # }, leg='L')

#error 23

#=============Procesamiento de sentadillas==================
for participante in participantes[:]: # llego hasta el 11 0-1 faltan 2-4    
    session_id = participante['session_id']
    usuario = participante["participant_id"]
    print(f'sesion_id {session_id}, usuario {usuario}')
    for movimiento in participante['movements']:
        if 'estocada_lateral_derecha' in movimiento['move'] or 'estocada_lateral_izquierda' in movimiento['move'] :

            move_name = movimiento['move']
            process_movement(usuario,session_id, movimiento)
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

guardar_resultados(reporte, f"errores_{name_file}_estlat.json")

print('tiempo total:', (time.time()-iniciar)/60)
print('correctos:', fines)
print('errores:', fails)
print("type_error:", len(es))
print('errors\n', es)


#=============Procesamiento de escalon==================

# for participante in participantes[:]: # llego hasta el 11 0-1 faltan 2-4    
#     session_id = participante['session_id']
#     usuario = participante["participant_id"]
#     print(f'sesion_id {session_id}, usuario {usuario}')
#     for movimiento in participante['movements']:
#         if 'escalon' in movimiento['move']:

#             move_name = movimiento['move']
#             if any(word in move_name for word in ['derecha', 'derecho']):
#                 process_movement(usuario,session_id, movimiento, leg='R')
#             elif any(word in move_name for word in ['izquierda', 'izquierdo']):
#                 process_movement(usuario,session_id, movimiento, leg='L')
#     print(f'Termino participante {usuario}')

# reporte = {
#     "resumen": {
#         "total_movimientos": len(fines) + len(fails),
#         "exitosos": len(fines),
#         "fallidos": len(fails),
#         "errores_unicos": len(set(es))
#     },
    
#     "movimientos_fallidos": fails_datail,
# }

# guardar_resultados(reporte, f"errores_{name_file}_escalon.json")
# print('tiempo total:', (time.time()-iniciar)/60)
# print('correctos:', fines)
# print('errores:', fails)
# print("type_error:", len(es))
# print('errors\n', es)