'''
    ---------------------------------------------------------------------------
    OpenCap processing: mainOpenSimAD_simplified.py
    ---------------------------------------------------------------------------
    Versión optimizada para calcular únicamente fuerzas de reacción del piso
    
    Esta versión simplificada elimina la complejidad innecesaria del código original
    y se enfoca únicamente en calcular las Ground Reaction Forces (GRF).
'''

import os
import casadi as ca
import numpy as np
import sys
import yaml
import scipy.interpolate as interpolate
import platform
import copy
import pandas as pd
from utils import (storage_to_numpy, storage_to_dataframe, 
                download_kinematics, import_metadata, numpy_to_storage)
from utilsOpenSimAD import generateExternalFunction, import_metadata
def run_grf_calculation(baseDir, dataDir, subject, settings, case='0'):
    """
    Función optimizada para calcular únicamente las fuerzas de reacción del piso.
    
    Parámetros:
    - baseDir: Directorio base del proyecto
    - dataDir: Directorio de datos
    - subject: ID del sujeto
    - settings: Configuración simplificada
    - case: Caso de análisis
    """
    
    # %% Configuración simplificada
    # Solo mantener configuraciones esenciales para GRF
    

    # Paso necesario: generar función externa
    metadata = import_metadata(os.path.join(dataDir, subject, 'sessionMetadata.yaml'))
    OpenSimModel = metadata['openSimModel']

    

    trialName = settings['trial_name']

    generateExternalFunction(
        baseDir=baseDir,
        dataDir=dataDir,
        subject=subject,
        OpenSimModel=OpenSimModel,
        overwrite=True,
        treadmill=False,
        contact_side=settings.get('contact_side', 'all'),
        useExpressionGraphFunction=True
    )

    timeIntervals = settings['timeInterval']

    pathMotionFile = os.path.join(dataDir, 'OpenSimData', 'Kinematics',
                                trialName + '.mot')
    motion_file = storage_to_numpy(pathMotionFile)
    # If no time window is specified, use the whole motion file.
    if not timeIntervals:            
        timeIntervals = [motion_file['time'][0], motion_file['time'][-1]]
    

    timeElapsed = timeIntervals[1] - timeIntervals[0]
    
    # Reducir densidad de malla para mayor velocidad
    meshDensity = 50  # Reducido de 100 a 50
    N = int(round(timeElapsed * meshDensity, 2))
    
    # Discretización temporal
    tgrid = np.linspace(timeIntervals[0], timeIntervals[1], N+1)
    tgridf = np.zeros((1, N+1))
    tgridf[:,:] = tgrid.T
    
    # Configuración simplificada del modelo
    OpenSimModel = 'LaiUhlrich2022'
    if 'OpenSimModel' in settings:  
        OpenSimModel = settings['OpenSimModel']
    model_full_name = OpenSimModel + "_scaled_adjusted"
    
    # Solo incluir coordenadas esenciales para GRF
    withMTP = False  # Desactivar MTP para simplificar
    withArms = False  # Desactivar brazos para simplificar
    withLumbarCoordinateActuators = False  # Desactivar lumbar para simplificar
    
    # %% Rutas simplificadas
    pathMain = os.getcwd()
    pathSubjectData = os.path.join(dataDir, subject)
    pathOSData = os.path.join(pathSubjectData, 'OpenSimData')
    pathModelFolder = os.path.join(pathOSData, 'Model')
    pathModelFile = os.path.join(pathModelFolder, model_full_name + ".osim")
    pathExternalFunctionFolder = os.path.join(pathModelFolder, 'ExternalFunction')
    pathIKFolder = os.path.join(pathOSData, 'Kinematics')
    pathResults = os.path.join(pathOSData, 'Dynamics', trialName)
    os.makedirs(pathResults, exist_ok=True)
    
    # %% Coordenadas simplificadas
    # Solo coordenadas esenciales para GRF
    joints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
              'pelvis_ty', 'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l',
              'hip_rotation_l', 'hip_flexion_r', 'hip_adduction_r',
              'hip_rotation_r', 'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
              'subtalar_angle_r']
    
    nJoints = len(joints)
    
    # Coordenadas de pelvis para fuerzas
    groundPelvisJoints = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                          'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    groundPelvisJointsForces = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    
    # %% Cargar datos de cinemática
    from utilsOpenSimAD import getIK, filterDataFrame, interpolateDataFrame
    pathIK = os.path.join(pathIKFolder, trialName + '.mot')
    Qs_fromIK = getIK(pathIK, joints)
    
    # Filtrado simplificado
    Qs_fromIK_filter = filterDataFrame(Qs_fromIK, cutoff_frequency=30)
    Qs_toTrack = interpolateDataFrame(Qs_fromIK_filter, timeIntervals[0], timeIntervals[1], N)
    
    # %% Función externa simplificada
    # Solo necesitamos la función externa para calcular GRF
    F_name = 'F'
    dim = 3*nJoints
    F_map = np.load(os.path.join(pathExternalFunctionFolder, F_name + '_map.npy'), 
                    allow_pickle=True).item()
    
    print("Coordenadas incluidas en F_map['residuals']:", F_map['residuals'].keys())
    
    print("Dimensión usada actualmente:", dim)

    # Usar función de expresión gráfica para mayor velocidad
    from utilsOpenSimAD import getF_expressingGraph
    sys.path.append(pathExternalFunctionFolder)
    os.chdir(pathExternalFunctionFolder)
    print('Antes de getF_expressingGraph ')
    print(dim, F_name)   
    F = getF_expressingGraph(dim, F_name)
    print('Después de getF_expressingGraph ')
    sys.path.remove(pathExternalFunctionFolder)
    os.chdir(pathMain)
    
    # Cargar mapeo de la función
    F_map = np.load(os.path.join(pathExternalFunctionFolder, F_name + '_map.npy'), 
                    allow_pickle=True).item()
    
    # %% Configuración de esferas de contacto
    nContactSpheres = F_map['GRFs']['nContactSpheres']
    contactSpheres = {}
    contactSpheres['right'] = F_map['GRFs']['rightContactSpheres']
    contactSpheres['left'] = F_map['GRFs']['leftContactSpheres']    
    contactSpheres['all'] = contactSpheres['right'] + contactSpheres['left']
    
    contactSides = []
    if contactSpheres['right']:
        contactSides.append('right')
    if contactSpheres['left']:
        contactSides.append('left')
    
    # %% Índices para GRF
    idxGroundPelvisJointsinF = [F_map['residuals'][joint] for joint in groundPelvisJoints]    
    idxJoints4F = [joints.index(joint) for joint in list(F_map['residuals'].keys())]
    
    # %% Cálculo directo de GRF
    print('Calculando fuerzas de reacción del piso...')
    
    # Preparar datos de entrada
    QsQds_opt_nsc = np.zeros((nJoints*2, N+1))
    QsQds_opt_nsc[::2, :] = Qs_toTrack.to_numpy()[:,1::].T
    QsQds_opt_nsc[1::2, :] = np.gradient(Qs_toTrack.to_numpy()[:,1::].T, axis=1) / (timeElapsed/N)
    
    # Calcular aceleraciones
    Qdds_opt_nsc = np.gradient(QsQds_opt_nsc[1::2, :], axis=1) / (timeElapsed/N)
    
    # Extraer GRF para cada punto temporal
    GRF_all_opt = {}
    GRF_all_opt['all'] = np.zeros((len(contactSides)*3, N))
    
    for k in range(N):
        # Llamar función externa
        Tk = F(ca.vertcat(QsQds_opt_nsc[:, k], Qdds_opt_nsc[idxJoints4F, k]))
        
        # Extraer GRF
        for c_s, side in enumerate(contactSides):
            if c_s == 0:  # Primera iteración
                GRF_all_opt[side] = np.zeros((3, N))
            
            # Obtener índices de GRF para este lado
            idx_grf_side = F_map['GRFs'][side]
            GRF_all_opt[side][:, k] = Tk[idx_grf_side].full().flatten()
            GRF_all_opt['all'][c_s*3:(c_s+1)*3, k] = GRF_all_opt[side][:, k]
    
    # %% Guardar resultados
    results = {
        'GRF': GRF_all_opt,
        'time': tgridf.flatten(),
        'contact_sides': contactSides,
        'settings': settings
    }
    
    np.save(os.path.join(pathResults, f'grf_results_{case}.npy'), results)
    
    # %% Crear archivo de salida para OpenSim
    if 'write_output' in settings and settings['write_output']:
        from utils import numpy_to_storage
        
        # Preparar datos para archivo .mot
        labels = ['time']
        for side in contactSides:
            for dim in ['x', 'y', 'z']:
                labels.append(f'ground_force_{side}_v{dim}')
        
        data = np.zeros((N, len(labels)))
        data[:, 0] = tgridf.flatten()[:-1]  # Excluir último punto
        
        col_idx = 1
        for side in contactSides:
            for dim_idx, dim in enumerate(['x', 'y', 'z']):
                data[:, col_idx] = GRF_all_opt[side][dim_idx, :]
                col_idx += 1
        
        output_path = os.path.join(pathResults, f'GRF_{trialName}_{case}.mot')
        numpy_to_storage(labels, data, output_path, datatype='GRF')
        print(f'Resultados guardados en: {output_path}')
    
    print('Cálculo de GRF completado exitosamente.')
    return results

def create_simplified_settings(trial_name, time_interval, motion_type='other'):
    """
    Crear configuración simplificada para cálculo de GRF.
    """
    settings = {
        'trial_name': trial_name,
        'timeInterval': time_interval,
        'OpenSimModel': 'LaiUhlrich2022',
        'write_output': True,
        'meshDensity': 50,  # Reducido para mayor velocidad
        'filter_Qs_toTrack': True,
        'cutoff_freq_Qs': 30,
        'ipopt_tolerance': 4
    }
    
    return settings 