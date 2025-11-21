import json
import time
from funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from funtion_clean_forceplates0 import IntegrateForcepalte_vc0

# from data_participantes import participantes
def buscar_valor(lista, clave, valor_buscado):
    """
    Busca un valor dentro de una lista de diccionarios.

    :param lista: Lista de diccionarios.
    :param clave: Clave dentro del diccionario donde se buscará el valor.
    :param valor_buscado: Valor que se desea encontrar.
    :return: Lista de diccionarios que contienen el valor buscado.
    """
    resultados = [item for item in lista if item.get(clave) == valor_buscado]
    return resultados[0]

with open('participantes2.json', 'r') as file:
    participantes = json.load(file)

participante = participantes[0]
participante['movements']
movimiento = buscar_valor(participante['movements'], 'move', 'estocada_derecha_1')
print(f'Inicio del proceso ', participante['participant_id'])
# f

IntegrateForcepalte_legs(session_id=participante['session_id'],
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id=participante['participant_id'],
                legs='R')
print(f'Fin del proceso ', participante['participant_id'], participante['session_id'])


