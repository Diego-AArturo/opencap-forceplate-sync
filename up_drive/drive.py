import os
from googleapiclient import discovery
import json
from googleapiclient.http import MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools
import time
from googleapiclient.errors import HttpError

SCOPES = 'https://www.googleapis.com/auth/drive'

# Autenticación con Google Drive
store = file.Storage('storage.json')
creds = store.get()
if not creds or creds.invalid:
    flow = client.flow_from_clientsecrets('client_secret_32173353459-a0o57bi8dhn9b199k3nhqcl7g79bv153.apps.googleusercontent.com.json', SCOPES)
    creds = tools.run_flow(flow, store)
DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))

# Ruta relativa para acceder a la carpeta "Data" desde "up_drive"


drive_parent_folder_id = "1y-mq2hjP_sUjIN4yd9BFon2PfLolpDhS"  # Reemplaza con el ID de la carpeta en Drive

def create_drive_folder(folder_name, parent_id=None):
    """ Crea una carpeta en Google Drive y devuelve su ID """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = DRIVE.files().list(q=query, fields="files(id)").execute()
    folders = results.get('files', [])
    
    if folders:
        return folders[0]['id']  # Retorna el ID si ya existe
    
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id] if parent_id else []
    }
    folder = DRIVE.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

def upload_file(file_path, folder_id, retries=3):
    """ Sube un archivo a una carpeta de Google Drive con reintentos automáticos """
    file_name = os.path.basename(file_path)
    
    # Paso 1: Buscar si ya existe un archivo con ese nombre en la carpeta destino
    query = f"name = '{file_name}' and '{folder_id}' in parents and trashed = false"
    existing_files = DRIVE.files().list(q=query, fields="files(id)").execute().get("files", [])

    media = MediaFileUpload(file_path, resumable=True)
    
    for attempt in range(retries):
        try:
            if existing_files:
                # Si ya existe, actualiza el archivo existente
                file_id = existing_files[0]['id']
                DRIVE.files().update(fileId=file_id, media_body=media).execute()
            else:
                # Si no existe, crea uno nuevo
                file_metadata = {
                    "name": file_name,
                    "parents": [folder_id]
                }
                DRIVE.files().create(body=file_metadata, media_body=media, fields="id").execute()
            return
        except HttpError as error:
            print(f"Error al subir el archivo {file_path}: {error}")
            if attempt < retries - 1:
                print(f"Reintentando... ({attempt + 1}/{retries})")
                time.sleep(2 ** attempt)  # Espera exponencial antes de reintentar
            else:
                print(f"Error persistente al subir el archivo {file_path} después de {retries} intentos.")
                raise

def upload_selected_files(local_folder, drive_folder_id):
    """ Busca y sube solo los archivos con 'syncd' en el nombre, sin crear carpetas vacías """
    for subfolder in os.listdir(local_folder):
        subfolder_path = os.path.join(local_folder, subfolder)

        if os.path.isdir(subfolder_path):  # Solo revisamos carpetas
            files_to_upload = [f for f in os.listdir(subfolder_path) if "syncd" in f and os.path.isfile(os.path.join(subfolder_path, f))]
            
            if files_to_upload:  # Solo crea la carpeta en Drive si hay archivos para subir
                # print(f"Procesando carpeta: {subfolder}")
                
                for file_name in files_to_upload:
                    file_path = os.path.join(subfolder_path, file_name)
                    # print(f"Subiendo archivo: {file_name}")
                    upload_file(file_path, drive_folder_id)

def upload_files_to_correct_drive_folder(local_folder, drive_movement_folders, drive_part_folder_id):
    for movimiento_local in os.listdir(local_folder):
        local_mov_path = os.path.join(local_folder, movimiento_local)
        if not os.path.isdir(local_mov_path):
            continue
        
        movimiento_drive = movimiento_map.get(movimiento_local)
        if not movimiento_drive:
            # print(f"No se encontró mapeo para: {movimiento_local}")
            continue

        drive_folder_id = drive_movement_folders.get(movimiento_drive)
        if not drive_folder_id:
            drive_folder_id = create_drive_folder(movimiento_drive, drive_part_folder_id)
            drive_movement_folders[movimiento_drive] = drive_folder_id  # opcional: actualizar el dict
            # print(f"Carpeta creada en Drive: {movimiento_drive}")

        
        files_to_upload = [f for f in os.listdir(local_mov_path) if "syncd" in f]
        for file_name in files_to_upload:
            file_path = os.path.join(local_mov_path, file_name)
            # print(f"Subiendo_fuerza {file_name} a {movimiento_drive}")
            upload_file(file_path, drive_folder_id)

def upload_kinematic_files(local_folder, drive_movement_folders, drive_part_folder_id):
    for file_name in os.listdir(local_folder):
        if not os.path.isfile(os.path.join(local_folder, file_name)):
            continue

        movimiento_drive = movimiento_map.get(os.path.splitext(file_name)[0])
        if not movimiento_drive:
            # print(f"No se encontró mapeo para archivo kinemático: {file_name}")
            continue

        drive_folder_id = drive_movement_folders.get(movimiento_drive)
        if not drive_folder_id:
            drive_folder_id = create_drive_folder(movimiento_drive, drive_part_folder_id)
            drive_movement_folders[movimiento_drive] = drive_folder_id
            # print(f"Carpeta creada en Drive: {movimiento_drive}")

        file_path = os.path.join(local_folder, file_name)
        # print(f"Subiendo archivo kinemático: {file_name} a {movimiento_drive}")
        upload_file(file_path, drive_folder_id)

def get_drive_subfolders(parent_id):
    """ Devuelve un dict con nombre: id de las subcarpetas de una carpeta en Drive """
    query = f"mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
    results = DRIVE.files().list(q=query, fields="files(id, name)").execute()
    return {f['name']: f['id'] for f in results.get('files', [])}

def guardar_resultados(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Ejecutar la subida
folders_id_name = {
    'P1' :  'part1'  ,
    'P2' :  'part2'  ,
    'P3' :  'part3'  ,
    'P4' :  'part4'  ,
    'P5' :  'part5'  ,
    'P6' :  'part6'  ,
    'P7' :  'part7'  ,
    'P8' :  'part8'  ,
    'P9' :  'part9'  ,
    'P10': 'part10' ,
    'P11': 'part11' ,
    'P12': 'part12' ,
    'P13': 'part13' ,
    'P14': 'part14' ,
    'P15': 'part15' ,
    'P16': 'part16' ,
    'P17': 'part17' ,
    'P18': 'part18' ,
    'P19': 'part19' ,
    'P20': 'part20' ,
    'P21': 'part21' ,
    'P22': 'part22' ,
    # 'P22N': 'part22',
    'P23': 'part23' ,
    'P24': 'part24' ,
    'P25': 'part25' ,
    'P26': 'part26' ,
    'P27': 'part27' ,
    'P28': 'part28' ,
    'P29': 'part29' ,
    'P30': 'part30',
    
}


movimiento_map = {
    # 'escalon_derecho_1': 'Escalon_derecho',
    # 'escalon_derecho_2': 'Escalon_derecho',
    # 'escalon_izquierdo_1': 'Escalon_izquierdo',
    # 'escalon_izquierdo_2': 'Escalon_izquierdo',

    # 'Desliza-lat-derecho': 'Estocada_Lat_Derecha_Deslizamiento',
    # 'Desliza-lat-derecho-2': 'Estocada_Lat_Derecha_Deslizamiento',
    # 'Desliza-lat-izquierda-1': 'Estocada_Lat_Izquierda_Deslizamiento',
    # 'Desliza-lat-izquierda-2': 'Estocada_Lat_Izquierda_Deslizamiento',
    # 'Desliza-post-derecha-1': 'Estocada_Posterior_Derecha_Deslizamiento',
    # 'Desliza-post-derecha-2': 'Estocada_Posterior_Derecha_Deslizamiento',
    # 'Desliza-post-izquierda-1': 'Estocada_Posterior_Izquierda_Deslizamiento',
    # 'Desliza-post-izquierda-2': 'Estocada_Posterior_Izquierda_Deslizamiento',

    # 'estocada_deslizamiento_lateral_derecho_1': 'Estocada_Lat_Derecha_Deslizamiento',
    # 'estocada_deslizamiento_lateral_derecho_2': 'Estocada_Lat_Derecha_Deslizamiento',
    # 'estocada_deslizamiento_lateral_izquierdo_1': 'Estocada_Lat_Izquierda_Deslizamiento',
    # 'estocada_deslizamiento_lateral_izquierdo_2': 'Estocada_Lat_Izquierda_Deslizamiento',
    # 'estocada_deslizamiento_posterior_derecho_1': 'Estocada_Posterior_Derecha_Deslizamiento',
    # 'estocada_deslizamiento_posterior_derecho_2': 'Estocada_Posterior_Derecha_Deslizamiento',
    # 'estocada_deslizamiento_posterior_izquierdo_1': 'Estocada_Posterior_Izquierda_Deslizamiento',
    # 'estocada_deslizamiento_posterior_izquierdo_2': 'Estocada_Posterior_Izquierda_Deslizamiento',
    'estocada_lateral_derecha_1': 'Estocada_Lat_Derecha',
    'estocada_lateral_derecha_2': 'Estocada_Lat_Derecha',
    'estocada_lateral_izquierda_1': 'Estocada_Lat_Izquierda',
    'estocada_lateral_izquierda_2': 'Estocada_Lat_Izquierda',
    'estocada_derecha_1': 'Estocada_Derecha',
    'estocada_derecha_2': 'Estocada_Derecha',
    'estocada_izquierda_1': 'Estocada_Izquierda',
    'estocada_izquierda_2': 'Estocada_Izquierda',

    # 'sentadilla_60_1': 'Sentadilla60',
    # 'sentadilla_60_2': 'Sentadilla60',
    # 'sentadilla_90_1': 'Sentadilla90',
    # 'sentadilla_90_2': 'Sentadilla90'
}

primal_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
ids = os.listdir(primal_folder)


if os.path.exists('realizados.json'):
        with open('realizados.json', 'r') as f:
            realized = json.load(f)
else:
    realized = []

for id in ids:
    
    if id in realized:
        continue
    part_folder = folders_id_name.get(id)
    if not part_folder:
        # print(f"No se encontró carpeta de Drive para {id}")
        continue
    
    drive_part_folder_id = create_drive_folder(part_folder, drive_parent_folder_id)
    drive_subfolders = get_drive_subfolders(drive_part_folder_id)
    local_measured_path = os.path.join(primal_folder, id, 'MeasuredForces')
    local_kinematic = os.path.join(primal_folder, id, 'OpenSimData', 'Kinematics')
    if not os.path.exists(local_measured_path):
        print(f"No se encontró carpeta 'MeasuredForces' para {id}")
        continue

    print(f"Procesando {id} → {part_folder}")
    upload_files_to_correct_drive_folder(local_measured_path, drive_subfolders,drive_part_folder_id)
    upload_kinematic_files(local_kinematic, drive_subfolders,drive_part_folder_id)

    realized.append(id)

    guardar_resultados(realized, 'realizados.json')

    




