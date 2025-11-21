# Estructura del Proyecto

Este documento describe la organización de carpetas del proyecto después de la reorganización.

## Estructura de Carpetas

```
ForcePlateIntegration/
├── scripts/                    # Scripts principales y ejemplos
│   ├── autocode.py
│   ├── autocode4val.py
│   ├── example_*.py            # Ejemplos de uso
│   ├── extract_*.py            # Scripts de extracción de datos
│   └── ...
│
├── src/                        # Código fuente organizado por funcionalidad
│   ├── forceplates/            # Funciones de integración de plataformas de fuerza
│   │   ├── funtion_*.py
│   │   ├── function_kinetics.py
│   │   └── ...
│   │
│   ├── opensim/                # Scripts relacionados con OpenSimAD
│   │   ├── mainOpenSimAD.py
│   │   ├── settingsOpenSimAD.py
│   │   └── ...
│   │
│   ├── utils/                  # Utilidades generales
│   │   ├── utils.py
│   │   ├── utilsAPI.py
│   │   ├── utilsKinematics.py
│   │   └── ...
│   │
│   ├── validation/             # Scripts de validación
│   │   ├── validation.py
│   │   └── validation_sync.py
│   │
│   └── plotting/               # Scripts de generación de gráficas
│       ├── generate_aggregate_plots.py
│       └── generate_publication_figures.py
│
├── notebooks/                  # Notebooks Jupyter
│   ├── analisis.ipynb
│   ├── data_analisis.ipynb
│   └── ...
│
├── config/                     # Archivos de configuración
│   ├── Setup_JointReaction.xml
│   └── ID_setup/
│
├── Data/                       # Datos del proyecto
│   ├── *.json                  # Archivos JSON de configuración
│   ├── *.xlsx                  # Archivos Excel
│   ├── *.mat                   # Archivos MATLAB
│   └── P*/                     # Datos por participante
│
├── output/                     # Resultados y salidas
│   ├── logs/                   # Archivos de log
│   └── OpenSimPipeline/         # Resultados de OpenSim
│
├── UtilsDynamicSimulations/    # Utilidades de simulaciones dinámicas (sin cambios)
│
└── readme.md                   # Documentación principal
```

## Cambios Realizados

### 1. Scripts Principales (`scripts/`)
- Scripts de ejemplo y ejecutables principales
- Scripts de procesamiento y extracción de datos
- Scripts de automatización

### 2. Código Fuente (`src/`)
Organizado por funcionalidad:
- **forceplates/**: Funciones específicas para integración de plataformas de fuerza
- **opensim/**: Scripts relacionados con simulaciones OpenSimAD
- **utils/**: Utilidades reutilizables (API, procesamiento, visualización)
- **validation/**: Scripts de validación y sincronización
- **plotting/**: Generación de gráficas y figuras

### 3. Datos (`Data/`)
- Archivos JSON de configuración
- Archivos Excel con datos procesados
- Archivos MATLAB (.mat)
- Carpetas por participante (P1, P2, etc.)

### 4. Configuración (`config/`)
- Archivos XML de configuración de OpenSim
- Setup files para diferentes análisis

### 5. Output (`output/`)
- Logs y archivos de salida
- Resultados de procesamiento
- Pipeline de OpenSim

## Imports Actualizados

Los imports han sido actualizados para reflejar la nueva estructura:

```python
# Antes
from funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from utils import storage_to_numpy
from mainOpenSimAD import run_tracking

# Después
from src.forceplates.funtion_integrate_forceplates_legs import IntegrateForcepalte_legs
from src.utils.utils import storage_to_numpy
from src.opensim.mainOpenSimAD import run_tracking
```

## Notas Importantes

1. **UtilsDynamicSimulations/**: Esta carpeta se mantiene sin cambios ya que contiene código del framework OpenCap-processing.

2. **Carpetas de Resultados**: Las carpetas `graficas/`, `graficas_gfr/`, `resultados_fuerza/`, etc. se mantienen en la raíz ya que contienen resultados generados.

3. **Archivos __init__.py**: Se han creado archivos `__init__.py` en las carpetas `src/` y subcarpetas para que funcionen como módulos de Python.

4. **Compatibilidad**: Los scripts en `scripts/` pueden necesitar ajustes en sus rutas si hacen referencia a archivos de datos o configuración.

