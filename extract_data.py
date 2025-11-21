import os
import sys
sys.path.append("./../../")
script_folder,_ = os.path.split(os.path.abspath(__file__))

import numpy as np
import pandas as pd
import opensim
# import requests
import matplotlib.pyplot as plt
import utils as ut
import utilsKinematics
import re

# Expresión regular para eliminar el número final del nombre del trial
def extract_movement_name(trial_name):
    return re.sub(r'_\d+$', '', trial_name)

def extract_data_t(session_id, trial_name, time):
    #'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38'
    #'escalon_derecho_1'
    data_folder = os.path.abspath(os.path.join(script_folder, 'Data', session_id))

    if not os.path.exists(os.path.join(data_folder, 'MarkerData')):
        _, model_name = ut.download_kinematics(session_id, folder=data_folder)
    else:
        model_name, _ = os.path.splitext(ut.get_model_name_from_metadata(data_folder))

    # Iniciar análisis cinemático
    kinematics = utilsKinematics.kinematics(data_folder, trial_name, 
                                            modelName=model_name, 
                                            lowpass_cutoff_frequency_for_coordinate_values=10)

    opencap_settings = ut.get_main_settings(data_folder, trial_name)
    opencap_metadata = ut.import_metadata(os.path.join(data_folder, 'sessionMetadata.yaml'))

    # Ruta del archivo .mot
    mot_file_path = os.path.join(data_folder, 'OpenSimData', 'Kinematics', f'{trial_name}.mot')

    # Cargar el archivo .mot utilizando OpenSim
    mot_table = opensim.TimeSeriesTable(mot_file_path)

    # Convertir la tabla a un DataFrame de pandas
    mot_df = pd.DataFrame(mot_table.getMatrix().to_numpy())
    mot_df.columns = mot_table.getColumnLabels()
    mot_df['time'] = mot_table.getIndependentColumn()

    # Reordenar las columnas para que 'time' sea la primera columna
    mot_df = mot_df[['time'] + [col for col in mot_df.columns if col != 'time']]
    df_mot = mot_df.round(3)
    col = ['time','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r',
        'hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l']
    plt.plot(df_mot['time'],df_mot['knee_angle_r'])
    plt.plot(df_mot['time'],df_mot['knee_angle_l'])
    plt.grid()
    plt.show()
    df_mot = df_mot[col]
    # Mostrar las primeras filas del DataFrame
    #print(df_mot[df_mot['time']==time])
    return df_mot[df_mot['time']==time]



data = [
        # {'session_id': 'da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38', 'participante': 'P1',
        # 'trial_names': [
        #     # 'escalon_derecho_1',
        #     # 'escalon_izquierdo_1',
        #     # 'estocada_deslizamiento_lateral_derecho_1',
        #     # 'estocada_deslizamiento_lateral_izquierdo_3',
        #     # 'estocada_deslizamiento_posterior_derecho_1',
        #     # 'estocada_deslizamiento_posterior_izquierdo_3',
        #     # 'estocada_derecha_1',
        #     # 'estocada_izquierda_2',
        #     # 'estocada_lateral_derecha_2',
        #     # 'estocada_lateral_izquierda_1',
        #     # 'sentadilla_60_2',
        #     'sentadilla_90_1'],
        # 'times': [
        #     # [4.65, 6.5],  # escalón_derecho 4.65	6.5
        #     # [3.4, 5.4],  # escalón_izquierdo 3.4	5.4
        #     # [5.85, 7.817, 10.133],  # estocada_deslizamiento_lateral_derecho 5.85	7.817	10.133
        #     # [4.6, 5.6, 6.9],  # estocada_deslizamiento_lateral_izquierdo 4.6	5.6	6.9
        #     # [5.7, 7.3, 8.85],  # estocada_deslizamiento_posterior_derecho 5.7	7.3	8.85
        #     # [5.85, 7.883, 9.8],  # estocada_deslizamiento_posterior_izquierdo 5.85	7.883	9.8
        #     # [5.05, 7.6, 9.1],  # estocada_derecha 5.05	7.6	9.1
        #     # [3.55, 6.467, 9.05],  # estocada_izquierda 3.55	6.467	9.05
        #     # [6.467, 9.417, 13.333],  # estocada_lateral_derecha 6.467	9.417	13.333
        #     # [5.933, 9.467, 12.317],  # estocada_lateral_izquierda 5.933	9.467	12.317
        #     # [6.95, 12.217, 16.1],  # sentadilla_60 6.95	12.217	16.1
        #     [5.85, 10.933, 14.283]   # sentadilla_90  5.85	10.92	14.283
        # ]
        # },
        # {'session_id': 'a3724192-e2b6-4636-b176-3b3028d66230', 'participante': 'P2',
        # 'trial_names': [
        #     'escalon_derecho',
        #     'escalon_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_derecha
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '0199dfe7-2595-471e-8f49-d5a887434357', 'participante': 'P3',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'dd83a45d-85b0-4f5e-bd26-93d22c413ed9', 'participante': 'P4',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '34b7f090-1cbd-43ce-86a2-a40c50feec3f', 'participante': 'P5',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'bfd679e8-b5e3-4185-b2f2-011554045077', 'participante': 'P6',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '37ad96c7-d786-4338-81ea-0d58104e9bb5', 'participante': 'P7',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '1a934006-0010-471a-9c38-fbd7edb6ffbc', 'participante': 'P8',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '03142c4c-922a-4d9e-8cf3-a2eb229baa14', 'participante': 'P9',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'cb608973-5e67-4d57-8b64-2e73fbbc6361', 'participante': 'P10',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '39cb5ebb-153d-4aa0-ada7-a28c849eec67', 'participante': 'P11',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '3eee7aa7-cece-4ead-a822-554b95b05613', 'participante': 'P12',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '8e903c93-5357-46f2-bacb-d485842546c0', 'participante': 'P13',
        # # 'trial_names': [
        # #     'escalón_derecho',
        # #     'escalón_izquierdo',
        # #     'estocada_deslizamiento_lateral_derecho',
        # #     'estocada_deslizamiento_lateral_izquierdo',
        # #     'estocada_deslizamiento_posterior_derecho',
        # #     'estocada_deslizamiento_posterior_izquierdo',
        # #     'estocada_izquierda',
        # #     'estocada_lateral_derecha',
        # #     'estocada_lateral_izquierda',
        # #     'sentadilla_60',
        # #     'sentadilla_90'],
        # # 'times': [
        # #     [1.23, 2.34],  # escalón_derecho
        # #     [1.12, 2.22],  # escalón_izquierdo
        # #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        # #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        # #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        # #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        # #     [1.05, 2.15, 3.25],  # estocada_izquierda
        # #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        # #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        # #     [1.75, 2.85, 3.95],  # sentadilla_60
        # #     [1.30, 2.40, 3.50]   # sentadilla_90
        # # ]},
        # {'session_id': 'ad28c4e1-9f27-4554-9e19-1063888ab302', 'participante': 'P14',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '8cff7224-37bf-44d5-94d0-f7dfdea5bc36', 'participante': 'P15',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'fe7e294e-b199-4c16-b9ca-c8c7841c42ba', 'participante': 'P16',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '634a9945-6598-4478-aa6f-3e88bdb2ab07', 'participante': 'P17',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90_2'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '126c9b36-d8f4-4d25-83e3-cd3afbc04148', 'participante': 'P18',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '67a1f5a5-1723-4c3c-86a5-54a06a0eb878', 'participante': 'P19',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'a3ba3d30-d5fd-4727-b460-b4310339a852', 'participante': 'P20',
        # 'trial_names': [
        #     'escalon_derecho_2',
        #     'escalon_izquierdo_1',
        #     'estocada_deslizamiento_lateral_derecho_2',
        #     'estocada_deslizamiento_lateral_izquierdo_2',
        #     'estocada_deslizamiento_posterior_derecho_2',
        #     'estocada_deslizamiento_posterior_izquierdo_2',
        #     'estocada_izquierda_2',
        #     'estocada_derecha_2',
        #     'estocada_lateral_derecha_2',
        #     'estocada_lateral_izquierda_2',
        #     'sentadilla_60_2',
        #     'sentadilla_90_2'],
        # 'times': [
        #     [4, 4.6],  # escalon_derecho
        #     [4.4, 4.9],  # escalon_izquierdo
        #     [5.15, 6.5, 7.8],  # estocada_deslizamiento_lateral_derecho
        #     [4.9, 6.3, 7.75],  # estocada_deslizamiento_lateral_izquierdo
        #     [4.5, 6.8, 8.55],  # estocada_deslizamiento_posterior_derecho
        #     [4, 5.15, 6.8],  # estocada_deslizamiento_posterior_izquierdo
        #     [4.75, 7.25, 9],  # estocada_izquierda
        #     [4.6, 6.95, 8.75],  # estocada_derecha
        #     [5, 7.15, 9],  # estocada_lateral_derecha
        #     [4.8, 6.75, 8.8],  # estocada_lateral_izquierda
        #     [5.25, 8.2, 10.65],  # sentadilla_60
        #     [4.45, 7.95, 10.25]   # sentadilla_90
        # ]},
        # {'session_id': 'f8ab78be-81ee-451f-affa-80a56880f741', 'participante': 'P21',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'c71d092b-93ba-4dfb-afc3-57fb46ef0736', 'participante': 'P22',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '85faebd8-e10e-41a4-a0cd-4abea37e1948', 'participante': 'P23',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '136cdc8a-cdf8-4745-8d12-a4499bfffe30', 'participante': 'P24',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '8d1c6dc2-552c-401f-af3d-c11b6dcebe3d', 'participante': 'P25',
        # 'trial_names': [
        #     # 'escalón_derecho',
        #     # 'escalón_izquierdo',
        #     # 'estocada_deslizamiento_lateral_derecho',
        #     # 'estocada_deslizamiento_lateral_izquierdo',
        #     # 'estocada_deslizamiento_posterior_derecho',
        #     # 'estocada_deslizamiento_posterior_izquierdo',
        #     # 'estocada_izquierda',
        #     # 'estocada_lateral_derecha',
        #     # 'estocada_lateral_izquierda',
        #     # 'sentadilla_60',
        #     'sentadilla_90_2'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'eb09846b-b49e-46e9-be68-73c5e0eff109', 'participante': 'P26',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': 'd8c338ad-3686-41d7-b70c-186aeacdc7cf', 'participante': 'P27',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '5ebde49f-8966-4258-bc17-5228a3e41b2d', 'participante': 'P28',
        # 'trial_names': [
        #     'escalón_derecho',
        #     'escalón_izquierdo',
        #     'estocada_deslizamiento_lateral_derecho',
        #     'estocada_deslizamiento_lateral_izquierdo',
        #     'estocada_deslizamiento_posterior_derecho',
        #     'estocada_deslizamiento_posterior_izquierdo',
        #     'estocada_izquierda',
        #     'estocada_lateral_derecha',
        #     'estocada_lateral_izquierda',
        #     'sentadilla_60',
        #     'sentadilla_90'],
        # 'times': [
        #     [1.23, 2.34],  # escalón_derecho
        #     [1.12, 2.22],  # escalón_izquierdo
        #     [1.45, 2.55, 3.65],  # estocada_deslizamiento_lateral_derecho
        #     [1.10, 2.20, 3.30],  # estocada_deslizamiento_lateral_izquierdo
        #     [1.80, 2.90, 3.99],  # estocada_deslizamiento_posterior_derecho
        #     [1.15, 2.25, 3.35],  # estocada_deslizamiento_posterior_izquierdo
        #     [1.05, 2.15, 3.25],  # estocada_izquierda
        #     [1.50, 2.60, 3.70],  # estocada_lateral_derecha
        #     [1.20, 2.30, 3.40],  # estocada_lateral_izquierda
        #     [1.75, 2.85, 3.95],  # sentadilla_60
        #     [1.30, 2.40, 3.50]   # sentadilla_90
        # ]},
        # {'session_id': '4862ff70-cf69-4ec8-9402-8a8d88b881ef', 'participante': 'P29',
        # 'trial_names': [
        #     'escalon_derecho_2', #2   3.55	 4.85
        #     'escalon_izquierdo_2', #2	3.95	5
        #     'estocada_deslizamiento_lateral_derecho_1',# 1	6.4	7.8	8.95
        #     'estocada_deslizamiento_lateral_izquierdo_2', #2	4.3	5.95	6.95
        #     'estocada_deslizamiento_posterior_derecho_2',# 2	5.05	7.25	8.25
        #     'estocada_deslizamiento_posterior_izquierdo_1',# 1	4.85	6.25	7.55
        #     'estocada_derecha_2',# 2	  3.35	  6.1	 9.6
        #     'estocada_izquierda_1', #1	4.45	6.65	8.95
        #     'estocada_lateral_derecha_1',# 1	5.05	8.25	11.65
        #     'estocada_lateral_izquierda_2', #2	5.05	10.1	14.15
        #     'sentadilla_60_2',# 2	4.95	7.6	    10.35
        #     'sentadilla_90_2'],# 2	3.65	7	10.35
        # 'times': [
        #     [3.55, 4.85],  # escalón_derecho
        #     [3.95, 5],  # escalón_izquierdo
        #     [6.4, 7.8, 8.95],  # estocada_deslizamiento_lateral_derecho
        #     [4.3, 5.95, 6.95],  # estocada_deslizamiento_lateral_izquierdo
        #     [5.05, 7.25, 8.25],  # estocada_deslizamiento_posterior_derecho
        #     [4.85, 6.25, 7.55],  # estocada_deslizamiento_posterior_izquierdo
        #     [3.35, 6.1,  9.6],  # estocada_derecha
        #     [4.45,	6.65,	8.95],  # estocada_izquierda
        #     [5.05,	8.25,	11.65],  # estocada_lateral_derecha
        #     [5.05,	10.1,	14.1],  # estocada_lateral_izquierda
        #     [4.95,	7.6,	    10.35],  # sentadilla_60
        #     [3.65,	7,	10.35]   # sentadilla_90
        # ]},
        # {'session_id': '86713dcf-995b-40e1-8df2-5da494cb170b', 'participante': 'P30',
        # 'trial_names': [
        #     'escalon_derecho_1',# 1	3.85	5.05
        #     'escalon_izquierdo_2',# 2	3.85	5.05
        #     'estocada_deslizamiento_lateral_derecho_1', #1	5.85	7.25	8.55
        #     'estocada_deslizamiento_lateral_izquierdo_2',# 2 	4.55	6.35	7.4
        #     'estocada_deslizamiento_posterior_derecho_1',# 1   4.3	 6.2	  7.75
        #     'estocada_deslizamiento_posterior_izquierdo_1',# 1	4.85	7.1	  8.2
        #     'estocada_derecha_1',# 1	  4.65	 8.7 	12.45
        #     'estocada_izquierda_1',# 1	3.95	6.4 	8.45
        #     'estocada_lateral_derecha_2',# 2	 5.6	11.2	14.45
        #     'estocada_lateral_izquierda_1',# 1	5.65	8.15	10.45
        #     'sentadilla_60_1',# 1	4.9	 8.25	10.85
        #     'sentadilla_90_1'],# 1	4.55	9.2	 12.35
        # 'times': [
        #     [3.85,	5.05],  # escalón_derecho
        #     [3.85,	5.05],  # escalón_izquierdo
        #     [5.85,	7.25,	8.55],  # estocada_deslizamiento_lateral_derecho
        #     [4.55,	6.35,	7.4],  # estocada_deslizamiento_lateral_izquierdo
        #     [4.3,	 6.2,	  7.75],  # estocada_deslizamiento_posterior_derecho
        #     [4.85,	7.1,	  8.2],  # estocada_deslizamiento_posterior_izquierdo
        #     [4.65,	 8.7, 	12.45],  # estocada_derecha
        #     [3.95,	6.4, 	8.45],  # estocada_izquierda
        #     [5.6,	11.2,	14.45],  # estocada_lateral_derecha
        #     [5.65,	8.15,	10.45],  # estocada_lateral_izquierda
        #     [4.9,	 8.25,	10.85],  # sentadilla_60
        #     [4.55,	9.2,	 12.35]   # sentadilla_90
        # ]},
        
]

#========================EXTRAER DATOS UNO X UNO ====================================
# datf = extract_data_t(session_id='da9fcf0a-3ec2-4990-b1e5-7f0d8dec5d38', trial_name='escalon_derecho_1', time=4.65)

# data_for_frame = {
#     'time':datf['time'],
#     'F1-CADSAGITAL I': datf['hip_flexion_l'],
#     'F1-CADSAGITAL D': datf['hip_flexion_r'], 
#     'F1-CADFRONTAL I': datf['hip_adduction_l'],
#     'F1-CADFRONTAL D': datf['hip_adduction_r'],
#     'F1-CADTRANS I': datf['hip_rotation_l'],
#     'F1-CADTRANS D': datf['hip_rotation_r'],
#     'F1-RODSAGITAL I': datf['knee_angle_l'],
#     'F1-RODSAGITAL D': datf['knee_angle_r'],
#     'F1-TOBILLOSAGITAL I': datf['ankle_angle_l'],
#     'F1-TOBILLOSAGITAL D': datf['ankle_angle_r'],
#     'F1-TOBILLOFRONTAL I': datf['subtalar_angle_l'],
#     'F1-TOBILLOFRONTAL D': datf['subtalar_angle_r'],
# }

# df = pd.DataFrame(data_for_frame)
# print(df)

#=====================para extraer los datos de los participantes=========================
movement_dfs = {extract_movement_name(trial): [] for trial in data[0]['trial_names']}

# Procesar los datos
for participant in data:
    session_id = participant['session_id']
    for trial_name, times in zip(participant['trial_names'], participant['times']):
        base_trial_name = extract_movement_name(trial_name)  # Obtener el nombre base sin el número
        for time in times:
            try:
                datf = extract_data_t(session_id=participant['participante'], trial_name=trial_name, time=time)
                data_for_frame = {
                    'participant': participant['participante'],
                    'time': datf['time'].iloc[0] if not datf['time'].empty else None,
                    # 'F1-CADSAGITAL I': datf['hip_flexion_l'].iloc[0] if not datf['hip_flexion_l'].empty else None,
                    # 'F1-CADSAGITAL D': datf['hip_flexion_r'].iloc[0] if not datf['hip_flexion_r'].empty else None,
                    # 'F1-CADFRONTAL I': datf['hip_adduction_l'].iloc[0] if not datf['hip_adduction_l'].empty else None,
                    # 'F1-CADFRONTAL D': datf['hip_adduction_r'].iloc[0] if not datf['hip_adduction_r'].empty else None,
                    # 'F1-CADTRANS I': datf['hip_rotation_l'].iloc[0] if not datf['hip_rotation_l'].empty else None,
                    # 'F1-CADTRANS D': datf['hip_rotation_r'].iloc[0] if not datf['hip_rotation_r'].empty else None,
                    'F1-RODSAGITAL I': datf['knee_angle_l'].iloc[0] if not datf['knee_angle_l'].empty else None,
                    'F1-RODSAGITAL D': datf['knee_angle_r'].iloc[0] if not datf['knee_angle_r'].empty else None,
                    # 'F1-TOBILLOSAGITAL I': datf['ankle_angle_l'].iloc[0] if not datf['ankle_angle_l'].empty else None,
                    # 'F1-TOBILLOSAGITAL D': datf['ankle_angle_r'].iloc[0] if not datf['ankle_angle_r'].empty else None,
                    # 'F1-TOBILLOFRONTAL I': datf['subtalar_angle_l'].iloc[0] if not datf['subtalar_angle_l'].empty else None,
                    # 'F1-TOBILLOFRONTAL D': datf['subtalar_angle_r'].iloc[0] if not datf['subtalar_angle_r'].empty else None,
                }
                movement_dfs[base_trial_name].append(data_for_frame)
            except Exception as e:
                print(f"Error processing {trial_name} at time {time} for participant {participant['participante']}: {e}")

abreviaciones = {
    "escalon_derecho": "ESCALON D",
    "escalon_izquierdo": "ESCALON I",
    "estocada_deslizamiento_lateral_derecho": "EST.DESl.LATERAL D",
    "estocada_deslizamiento_lateral_izquierdo": "EST.DESl.LATERAL I",
    "estocada_deslizamiento_posterior_derecho": "EST.DESL.POSTERIOR D",
    "estocada_deslizamiento_posterior_izquierdo": "EST.DESL.POSTERIOR I",
    "estocada_derecha": "EST.POSTERIOR D",
    "estocada_izquierda": "EST.POSTERIOR I",
    "estocada_lateral_derecha": "EST.LATERAL D",
    "estocada_lateral_izquierda": "EST.LATERAL I",
    "sentadilla_60": "SENT 60",
    "sentadilla_90": "SENT 90"
}
# Guardar en un archivo Excel con múltiples hojas
with pd.ExcelWriter('movements_data.xlsx') as writer:
    for trial_name, data_list in movement_dfs.items():
        df = pd.DataFrame(data_list)
        sheet_name = abreviaciones.get(trial_name, trial_name)

        # Asegurar que el nombre de la hoja tenga máximo 31 caracteres
        sheet_name = sheet_name[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)