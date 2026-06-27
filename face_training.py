"""
face_training.py - Entrenamiento del modelo EigenFaces
Lee todas las imágenes de DataFaces/ y entrena el modelo.
MEJORA: usa preprocesar_rostro() idéntico al reconocimiento para consistencia.
"""

import cv2
import os
import numpy as np
import json
from utils import DATA_DIR, MODEL_PATH, listar_usuarios, preprocesar_rostro


def entrenar_modelo(callback_log=None) -> dict:
    """
    Entrena el modelo EigenFaces con las imágenes registradas.
    Usa el mismo preprocesamiento (CLAHE + normalización) que el reconocedor,
    lo cual es crítico para que la predicción sea correcta.
    """
    def log(msg):
        print(msg)
        if callback_log:
            callback_log(msg)

    usuarios = listar_usuarios()
    if not usuarios:
        return {
            "exito": False,
            "error": "No hay estudiantes registrados. Registra al menos uno primero."
        }

    labels      = []
    faces_data  = []
    label_map   = {}   # índice → nombre

    log(f"Usuarios encontrados: {usuarios}")

    for label_idx, nombre in enumerate(sorted(usuarios)):   # orden determinista
        user_path = os.path.join(DATA_DIR, nombre)
        label_map[label_idx] = nombre
        log(f"  → Cargando imágenes de: {nombre}")
        cargadas = 0

        for filename in sorted(os.listdir(user_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(user_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # ── MISMO preprocesamiento que en reconocimiento ──────────────────
            img_proc = preprocesar_rostro(img)
            faces_data.append(img_proc)
            labels.append(label_idx)
            cargadas += 1

        log(f"     {cargadas} imágenes cargadas para '{nombre}'.")

    if len(faces_data) == 0:
        return {"exito": False, "error": "No se encontraron imágenes válidas."}

    if len(set(labels)) < 2:
        return {
            "exito": False,
            "error": "EigenFaces necesita al menos 2 estudiantes distintos para entrenar."
        }

    log("Entrenando modelo EigenFaces...")
    try:
        recognizer = cv2.face.EigenFaceRecognizer_create(
            num_components=0,   # 0 = automático (min(N_muestras, N_píxeles))
        )
    except AttributeError as e:
        return {
            "exito": False,
            "error": f"OpenCV no tiene el módulo face ({e}). Instala opencv-contrib-python en lugar de opencv-python."
        }
    recognizer.train(faces_data, np.array(labels))

    log(f"Guardando modelo en: {MODEL_PATH}")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.write(MODEL_PATH)

    # Guardar mapa etiqueta → nombre
    label_map_path = MODEL_PATH.replace(".xml", "_labels.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    log(f"Entrenamiento completado: {len(faces_data)} imágenes, {len(usuarios)} usuarios.")

    return {
        "exito":          True,
        "usuarios":       sorted(usuarios),
        "label_map":      label_map,
        "total_imagenes": len(faces_data),
    }


if __name__ == "__main__":
    print("=== Entrenamiento del Modelo ===")
    resultado = entrenar_modelo()
    if resultado["exito"]:
        print(f"\nModelo listo con {resultado['total_imagenes']} imágenes.")
        print(f"   Usuarios: {resultado['usuarios']}")
    else:
        print(f"\nError: {resultado.get('error')}")
