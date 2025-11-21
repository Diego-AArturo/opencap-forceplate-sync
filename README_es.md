# Integración de Plataformas de Fuerza con OpenCap

## Descripción General

Este proyecto proporciona herramientas para sincronizar datos de plataformas de fuerza con datos cinemáticos de OpenCap. El objetivo principal es integrar mediciones temporales y espaciales de plataformas de fuerza con el sistema de captura de movimiento basado en marcadores de OpenCap, permitiendo análisis biomecánicos completos.

## Repositorio

**Repositorio del proyecto**: [https://github.com/Diego-AArturo/opencap-forceplate-sync](https://github.com/Diego-AArturo/opencap-forceplate-sync)

Este proyecto está basado en y extiende las funcionalidades del repositorio original:

**Repositorio original**: [https://github.com/stanfordnmbl/opencap-processing](https://github.com/stanfordnmbl/opencap-processing)  
**Desarrollado por**: Stanford Neuromuscular Biomechanics Laboratory

Agradecemos al Stanford Neuromuscular Biomechanics Laboratory por proporcionar las herramientas base de procesamiento de OpenCap que hicieron posible este proyecto.

## Objetivo Principal

El objetivo principal de este sistema es **sincronizar datos cinemáticos de OpenCap con mediciones de plataformas de fuerza**, permitiendo a los investigadores combinar:
- **Datos cinemáticos** de OpenCap (posiciones de marcadores, ángulos articulares, etc.)
- **Datos cinéticos** de plataformas de fuerza (fuerzas de reacción del suelo, momentos, centro de presión)

Esta sincronización permite análisis biomecánicos completos incluyendo cálculos de dinámica inversa en OpenSim.

## Inicio Rápido

El script principal a ejecutar es **`scripts/batch_process_forceplates.py`**, que procesa múltiples participantes y movimientos automáticamente.

## Flujo de Trabajo Paso a Paso

### Paso 1: Preprocesar Archivos de Plataformas de Fuerza

Antes del procesamiento, los archivos de plataformas de fuerza deben ser preprocesados usando el notebook de Jupyter:

**`notebooks/plataformas_fuerza.ipynb`**

Este notebook:
- Procesa archivos de datos crudos de plataformas de fuerza
- Estandariza las convenciones de nombres de columnas
- Maneja casos especiales (por ejemplo, movimientos de deslizamiento lateral requieren intercambio de etiquetas de piernas)
- Prepara archivos para la integración

**Importante**: Para movimientos de deslizamiento lateral (`deslizamiento_lateral`), el sistema automáticamente intercambia las etiquetas de pierna derecha/izquierda (`r_` ↔ `L_`, `l_` ↔ `R_`) para coincidir con el patrón de movimiento.

### Paso 2: Crear Archivo JSON de Configuración de Participantes

Después del preprocesamiento, cree un archivo JSON que relacione los datos de plataformas de fuerza con las sesiones de video de OpenCap. Los nombres de movimiento en el JSON **deben coincidir exactamente** con los nombres de prueba en OpenCap.

**Estructura del Archivo** (ejemplo: `participantes.json`):

```json
[
    {
        "participant_id": "P1",
        "session_id": "opencap_session_id_aqui",
        "movements": [
            {
                "move": "nombre_movimiento",
                "link": "url_google_drive_o_vacio"
            },
            {
                "move": "nombre_movimiento_2",
                "link": "url_google_drive_o_vacio"
            }
        ]
    },
    {
        "participant_id": "P2",
        "session_id": "otro_opencap_session_id",
        "movements": [
            {
                "move": "nombre_movimiento",
                "link": "url_google_drive_o_vacio"
            }
        ]
    }
]
```

**Requisitos Clave**:
- `participant_id`: Identificador único para el participante
- `session_id`: ID de sesión de OpenCap (encontrado en la interfaz web de OpenCap)
- `move`: **Debe coincidir exactamente** con el nombre de prueba en OpenCap
- `link`: URL de Google Drive para datos de plataformas de fuerza (puede estar vacío si se usan archivos locales)

### Paso 3: Configurar y Ejecutar Procesamiento por Lotes

1. Edite `scripts/batch_process_forceplates.py`:
   - Actualice la variable `name_file` para que coincida con el nombre de su archivo JSON (sin la extensión `.json`)
   - Configure los conjuntos de clasificación de movimientos (`unprocessed`, `bothlegs`) si es necesario

2. Ejecute el script:
   ```bash
   python scripts/batch_process_forceplates.py
   ```

3. El script:
   - Cargará datos de participantes desde el archivo JSON
   - Descargará o cargará datos de plataformas de fuerza
   - Descargará datos cinemáticos de OpenCap (si no están presentes)
   - Sincronizará coordenadas temporales y espaciales
   - Ejecutará análisis de dinámica inversa en OpenSim
   - Generará gráficos de sincronización

### Paso 4: Acceder a Datos Sincronizados

Después del procesamiento, los datos sincronizados se pueden encontrar en:

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

## Métodos de Integración

El sistema proporciona dos métodos de integración dependiendo del tipo de movimiento y la información disponible:

### Método 1: Integración de Pierna Única
**Archivo**: `src/forceplates/funtion_integrate_forceplates_legs.py`  
**Función**: `IntegrateForcepalte_legs()`

**Usar cuando**:
- Solo un pie contacta la plataforma de fuerza
- Se sabe qué pie contacta la plataforma primero
- El movimiento es claramente unilateral (por ejemplo, paso unilateral, estocada unilateral)

**Parámetros**:
- `legs`: Especifique `'R'` para pierna derecha o `'L'` para pierna izquierda

**Ejemplo**:
```python
IntegrateForcepalte_legs(
    session_id=session_id,
    trial_name="escalon_derecho_1",
    force_gdrive_url=force_url,
    participant_id="P1",
    legs='R'  # Especifique la pierna
)
```

### Método 2: Integración de Ambas Piernas (Auto-Detección)
**Archivo**: `src/forceplates/integrate_forceplates_both_legs.py`  
**Función**: `IntegrateForcepalte_vc0()`

**Usar cuando**:
- Ambos pies contactan plataformas de fuerza simultáneamente
- No se sabe qué pie contactó primero
- El movimiento es bilateral (por ejemplo, sentadillas, saltos, estocadas bilaterales)

**Características**:
- Detecta automáticamente qué pierna inició el contacto primero
- Compara eventos de contacto del talón de ambas piernas
- Usa señales de fuerza combinadas para sincronización

**Ejemplo**:
```python
IntegrateForcepalte_vc0(
    session_id=session_id,
    trial_name="sentadilla_90_1",
    force_gdrive_url=force_url,
    participant_id="P1"
    # No se necesita parámetro de pierna - se detecta automáticamente
)
```

**Selección Automática**: El script de procesamiento por lotes (`batch_process_forceplates.py`) selecciona automáticamente el método apropiado basándose en patrones de nombres de movimiento:
- Movimientos con `'derecha'` o `'derecho'` → Pierna única (derecha)
- Movimientos con `'izquierda'` o `'izquierdo'` → Pierna única (izquierda)
- Movimientos en el conjunto `bothlegs` (por ejemplo, `'sentadilla'`) → Integración de ambas piernas

## Validación con Estándar de Oro

Para validar los resultados de sincronización contra un sistema estándar de oro (por ejemplo, captura de movimiento Motive con plataformas de fuerza sincronizadas por hardware), use los scripts de validación en la carpeta `src/validation/`.

### Script de Validación Rápida
**Archivo**: `src/validation/autocode4val.py`

Este script procesa múltiples pruebas y compara datos sincronizados de OpenCap con mediciones del estándar de oro.

### Validación Integral
**Archivo**: `src/validation/validation_sync.py`

La clase `SyncValidator` proporciona validación integral incluyendo:
- Comparación de alineación temporal
- Análisis de correlación de señales de fuerza
- Cálculos de RMSE y MAE
- Gráficos de visualización
- Reportes detallados de validación

**Ejemplo de Uso**:
```python
from src.validation.validation_sync import SyncValidator

validator = SyncValidator(
    gold_standard_mot_path="ruta/a/gold_forces.mot",
    gold_standard_trc_path="ruta/a/gold_markers.trc",
    opencap_syncd_mot_path="Data/P1/MeasuredForces/trial1/trial1_syncd_forces.mot",
    opencap_trc_path="Data/P1/MarkerData/trial1.trc",
    output_folder="validation_results"
)

metrics = validator.run_validation()
```

## Archivos de Salida

### Datos de Fuerza Sincronizados
- **Ubicación**: `Data/{participant_id}/MeasuredForces/{trial_name}/{trial_name}_syncd_forces.mot`
- **Formato**: Archivo MOT de OpenSim con datos sincronizados de plataformas de fuerza
- **Contenido**: Fuerzas de reacción del suelo, momentos y centro de presión para ambas piernas

### Resultados de Dinámica Inversa
- **Ubicación**: `Data/{participant_id}/OpenSimData/InverseDynamics/{trial_name}/{trial_name}.sto`
- **Formato**: Archivo STO de OpenSim
- **Contenido**: Momentos y potencias articulares calculados a partir de datos sincronizados

### Gráficos de Sincronización
- **Ubicación**: `graficas/{participant_id}/{trial_name}_corte.png`
- **Contenido**: Visualización de posición del talón y punto de sincronización

### Tiempos de Retraso (Lag Times)
- **Archivo**: `lag_times.json`
- **Contenido**: Valores de desplazamiento temporal calculados para cada prueba
- **Propósito**: Registra parámetros de sincronización para referencia

## Estructura del Proyecto

```
ForcePlateIntegration/
├── scripts/
│   └── batch_process_forceplates.py      # Script principal de procesamiento por lotes
├── src/
│   ├── forceplates/
│   │   ├── funtion_integrate_forceplates_legs.py    # Integración de pierna única
│   │   └── integrate_forceplates_both_legs.py      # Integración de ambas piernas
│   └── validation/
│       ├── validation_sync.py            # Validación integral
│       └── autocode4val.py                # Script de validación rápida
├── notebooks/
│   └── plataformas_fuerza.ipynb          # Preprocesamiento de plataformas de fuerza
├── Data/                                  # Directorio de salida
└── README_es.md                           # Este archivo
```

## Requisitos

Consulte `requirements.txt` para la lista completa de dependencias. Las dependencias clave incluyen:
- numpy
- pandas
- matplotlib
- scipy
- opensim
- requests

## Notas

- Los nombres de movimiento en la configuración JSON **deben coincidir exactamente** con los nombres de prueba en OpenCap
- Los archivos de plataformas de fuerza deben ser preprocesados antes de la integración
- El sistema maneja automáticamente las transformaciones del sistema de coordenadas
- La calibración espacial alinea los vectores de fuerza con marcadores anatómicos
- La sincronización temporal usa detección de contacto del talón y correlación cruzada

## Créditos y Agradecimientos

Este proyecto está basado en el repositorio de procesamiento de OpenCap desarrollado por el **Stanford Neuromuscular Biomechanics Laboratory**:

- **Repositorio original**: [https://github.com/stanfordnmbl/opencap-processing](https://github.com/stanfordnmbl/opencap-processing)
- **Laboratorio**: Stanford Neuromuscular Biomechanics Laboratory

Agradecemos al equipo de Stanford por proporcionar las herramientas base que hicieron posible este proyecto de integración de plataformas de fuerza.

## Soporte

Para problemas o preguntas, consulte la documentación del código o contacte al equipo de desarrollo.

---

**Autor**: Equipo ForcePlateIntegration  
**Repositorio**: [https://github.com/Diego-AArturo/opencap-forceplate-sync](https://github.com/Diego-AArturo/opencap-forceplate-sync)

