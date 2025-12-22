import json
import time
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
from src.forceplates.funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from src.forceplates.integrate_forceplates_both_legs import IntegrateForcepalte_vc0

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

with open('data_participantes\participantes4.json', 'r') as file:
    participantes = json.load(file)

participante = participantes[6]
participante['movements']
movimiento = buscar_valor(participante['movements'], 'move', 'estocada_deslizamiento_posterior_derecho_1')
print(f'Inicio del proceso ', participante['participant_id'])
# f

IntegrateForcepalte_vc0(session_id=participante['session_id'],
                trial_name=movimiento['move'],
                force_gdrive_url=movimiento['link'],
                participant_id=participante['participant_id'],
                )
print(f'Fin del proceso ', participante['participant_id'], movimiento['move'])





