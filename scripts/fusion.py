import c3d
import pandas as pd
import numpy as np



# ---------- CONFIGURACIÓN ----------
C3D_PATH = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\MOTIVE\sentadilla_90_1.c3d"
MOT_PATH = r"C:\Users\diegu\Documents\valle_fisio\Validacion 2\PLATAFORMAS_mod\sentadilla_90_1.mot"
OUTPUT_PATH = "sentadilla_60_1_combined.c3d"

# ---------- 1. CARGAR C3D (MARCADORES) ----------
with open(C3D_PATH, 'rb') as f:
    reader = c3d.Reader(f)
    frame_rate = reader.frame_rate
    n_frames = reader.point_count  # Número de frames
    point_labels = reader.point_labels

# ---------- 2. CARGAR MOT (FUERZAS) ----------
with open(MOT_PATH) as f:
    for i, line in enumerate(f):
        if line.strip().lower() == 'endheader':
            skip = i + 1
            break

mot = pd.read_csv(
    MOT_PATH,
    sep=r'\s+',
    skiprows=skip,
    engine='python'
)
analog_labels = mot.columns[1:]
analog_data = mot.iloc[:, 1:].to_numpy()
analog_frame_rate = 1000
analogs_per_frame = analog_frame_rate // int(frame_rate)

# ---------- 3. CONSTRUIR Y GUARDAR .C3D ----------
with open(OUTPUT_PATH, 'wb') as f:
    writer = c3d.Writer()
    writer.frame_rate = frame_rate
    writer.analog_sample_rate = analog_frame_rate
    writer.set_point_labels(point_labels)
    writer.set_analog_labels(list(analog_labels))
    
    # Leer el archivo original de nuevo para iterar frames
    with open(C3D_PATH, 'rb') as f_orig:
        reader_orig = c3d.Reader(f_orig)
        for frame_num in range(n_frames):
            # Obtener datos de marcadores del c3d original
            marker_data, analog_chunk_orig = reader_orig.get_frame(frame_num)
            # Obtener datos de fuerzas del mot
            analog_start = frame_num * analogs_per_frame
            analog_end = analog_start + analogs_per_frame
            analog_chunk = analog_data[analog_start:analog_end].T
            
            writer.add_frames(marker_data, analog_chunk)

    writer.write(f)

print(f"✅ Archivo exportado como: {OUTPUT_PATH}")

