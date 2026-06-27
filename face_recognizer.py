"""
face_recognizer.py - Reconocimiento facial + análisis de emoción y sueño
MEJORAS:
  - Preprocesamiento idéntico al entrenamiento (CLAHE + normalización)
  - Detección de sonrisa y tensión facial para 9 emociones
  - Suavizado temporal: promedia las últimas N predicciones para evitar saltos
  - Umbral de confianza calibrado correctamente
  - El nombre se asocia SIEMPRE desde el label_map del modelo entrenado
"""

import cv2
import os
import json
import numpy as np
from collections import deque

from utils import (
    cargar_clasificadores, cargar_modelo, listar_usuarios,
    calcular_porcentaje_sueno, analizar_emocion, calcular_motivacion,
    detectar_sonrisa, calcular_tension_facial, preprocesar_rostro,
    detectar_boca_abierta, evaluar_cejas, evaluar_reflejos_nariz,
    obtener_emociones_candidatas, guardar_registro,
    EMOCIONES, MODEL_PATH, MIN_CONFIDENCE
)

# Cuántos frames promediar para suavizar predicciones
SMOOTH_WINDOW = 8


def cargar_label_map():
    """Carga el mapa etiqueta → nombre desde JSON (generado al entrenar)."""
    label_map_path = MODEL_PATH.replace(".xml", "_labels.json")
    if not os.path.exists(label_map_path):
        # Fallback: orden alfabético de carpetas (igual que en entrenamiento)
        usuarios = sorted(listar_usuarios())
        return {str(i): u for i, u in enumerate(usuarios)}
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def dibujar_hud(frame, x, y, w, h, nombre, emocion, sueno, motivacion, confianza,
                 boca_abierta=False, cejas_intensas=0, reflejos_nariz=0):
    """Dibuja el HUD de información sobre el frame."""
    info_emocion = EMOCIONES.get(emocion, {})
    color_emocion = info_emocion.get("color", (200, 200, 200))

    # Recuadro del rostro
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_emocion, 2)

    # Panel semitransparente debajo del rostro
    overlay = frame.copy()
    panel_y = y + h + 5
    panel_h = 145
    panel_w = max(w + 90, 240)
    cv2.rectangle(overlay, (x, panel_y), (x + panel_w, panel_y + panel_h), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Nombre del estudiante (grande)
    cv2.putText(frame, nombre,
                (x + 6, panel_y + 24), font, 0.68, (255, 255, 255), 2)

    # Emoción con color característico
    cv2.putText(frame, emocion,
                (x + 6, panel_y + 46), font, 0.52, color_emocion, 1)

    # Métricas secundarias
    cv2.putText(frame, f"Sueño:      {sueno}%",
                (x + 6, panel_y + 68), font, 0.44, (150, 200, 255), 1)
    cv2.putText(frame, f"Motivación: {motivacion}%",
                (x + 6, panel_y + 84), font, 0.44, (100, 255, 150), 1)
    cv2.putText(frame, f"Conf: {confianza:.0f}",
                (x + 6, panel_y + 100), font, 0.38, (180, 180, 180), 1)

    boca_texto = "Abierta" if boca_abierta else "Cerrada"
    color_boca = (255, 215, 120) if boca_abierta else (180, 180, 180)
    cv2.putText(frame, f"Boca: {boca_texto}",
                (x + 6, panel_y + 118), font, 0.38, color_boca, 1)

    cejas_texto = f"Cejas: {cejas_intensas}%"
    color_cejas = (255, 150, 30) if cejas_intensas > 35 else (180, 180, 180)
    cv2.putText(frame, cejas_texto,
                (x + 120, panel_y + 118), font, 0.38, color_cejas, 1)

    nariz_texto = f"Nariz: {reflejos_nariz}%"
    color_nariz = (120, 255, 180) if reflejos_nariz > 22 else (180, 180, 180)
    cv2.putText(frame, nariz_texto,
                (x + 6, panel_y + 136), font, 0.38, color_nariz, 1)

    # Barra de motivación
    bar_x, bar_y = x + 6, panel_y + 148
    bar_w = panel_w - 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
    fill = int(bar_w * motivacion / 100)
    if motivacion > 65:
        bar_color = (0, 210, 90)
    elif motivacion > 40:
        bar_color = (0, 160, 255)
    else:
        bar_color = (50, 50, 240)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 10), bar_color, -1)

    return frame


def iniciar_reconocimiento(duracion_segundos=None, callback_frame=None,
                            callback_resultado=None) -> dict:
    """
    Inicia el reconocimiento facial en tiempo real.

    Mejoras clave:
    - preprocesar_rostro() = CLAHE + normalización (idéntico al entrenamiento)
    - Suavizado temporal: promedia últimas SMOOTH_WINDOW predicciones
    - Detección de sonrisa y tensión para 9 estados emocionales
    - Nombre siempre viene del label_map guardado al entrenar
    """
    try:
        modelo = cargar_modelo()
    except RuntimeError as e:
        return {"exito": False, "error": str(e)}

    if modelo is None:
        return {"exito": False, "error": "Modelo no entrenado. Entrena primero."}

    face_cascade, eye_cascade, mouth_cascade = cargar_clasificadores()
    label_map = cargar_label_map()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        return {"exito": False, "error": "No se pudo acceder a la cámara."}

    import time
    inicio = time.time()
    ultimo_resultado = {}
    registro_guardado = set()

    # Buffer de suavizado por persona detectada
    # Almacena deques de (label_id, confianza) para promediar
    historial_labels     = deque(maxlen=SMOOTH_WINDOW)
    historial_confianzas = deque(maxlen=SMOOTH_WINDOW)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            rostro_gray = gray[y:y+h, x:x+w]

            # ── Preprocesamiento idéntico al entrenamiento ────────────────────
            rostro_proc = preprocesar_rostro(rostro_gray)

            try:
                label_id, confianza = modelo.predict(rostro_proc)
            except Exception:
                continue

            # ── Suavizado temporal ────────────────────────────────────────────
            historial_labels.append(label_id)
            historial_confianzas.append(confianza)

            # Moda de los últimos N labels (quién aparece más veces)
            from collections import Counter
            label_suavizado = Counter(historial_labels).most_common(1)[0][0]
            confianza_suavizada = np.mean(historial_confianzas)

            # ── Determinar nombre ──────────────────────────────────────────────
            if confianza_suavizada < MIN_CONFIDENCE:
                nombre = label_map.get(str(label_suavizado),
                                       f"Estudiante_{label_suavizado}")
            else:
                nombre = "Desconocido"

            # ── Análisis multimodal ───────────────────────────────────────────
            sueno        = calcular_porcentaje_sueno(gray, (x, y, w, h), eye_cascade)
            hay_sonrisa   = detectar_sonrisa(gray, (x, y, w, h), mouth_cascade)
            boca_abierta = detectar_boca_abierta(gray, (x, y, w, h))
            cejas_intensas = evaluar_cejas(gray, (x, y, w, h))
            reflejos_nariz = evaluar_reflejos_nariz(gray, (x, y, w, h))
            tension      = calcular_tension_facial(gray, (x, y, w, h))

            emocion    = analizar_emocion(
                sueno, confianza_suavizada,
                hay_sonrisa=hay_sonrisa,
                tension_facial=tension,
                boca_abierta=boca_abierta,
                cejas_intensas=cejas_intensas,
                reflejos_nariz=reflejos_nariz,
            )
            motivacion = calcular_motivacion(emocion, sueno, tension)
            emociones_candidatas = obtener_emociones_candidatas(
                sueno, confianza_suavizada,
                hay_sonrisa=hay_sonrisa,
                tension_facial=tension,
                boca_abierta=boca_abierta,
                cejas_intensas=cejas_intensas,
                reflejos_nariz=reflejos_nariz,
            )

            # ── Dibujar HUD ───────────────────────────────────────────────────
            frame = dibujar_hud(frame, x, y, w, h, nombre, emocion,
                                 sueno, motivacion, confianza_suavizada,
                                 boca_abierta=boca_abierta,
                                 cejas_intensas=cejas_intensas,
                                 reflejos_nariz=reflejos_nariz)

            resultado = {
                "nombre":    nombre,
                "emocion":   emocion,
                "emociones_candidatas": emociones_candidatas,
                "sueno":     sueno,
                "motivacion": motivacion,
                "confianza": round(float(confianza_suavizada), 1),
                "sonrisa":   hay_sonrisa,
                "tension":   tension,
                "boca_abierta": boca_abierta,
                "cejas_intensas": cejas_intensas,
                "reflejos_nariz": reflejos_nariz,
            }
            ultimo_resultado = resultado

            if nombre != "Desconocido" and nombre not in registro_guardado:
                guardar_registro(nombre, emocion, sueno, motivacion)
                registro_guardado.add(nombre)

            if callback_resultado:
                callback_resultado(resultado)

        # Pie de pantalla
        cv2.putText(frame, "ESC: Detener | MotivaScan",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

        if callback_frame:
            callback_frame(frame)

        cv2.imshow("MotivaScan - Deteccion de Motivacion", frame)

        key = cv2.waitKey(10)
        if key == 27:
            break
        if duracion_segundos and (time.time() - inicio) > duracion_segundos:
            break

    webcam.release()
    cv2.destroyAllWindows()
    return {"exito": True, **ultimo_resultado}


if __name__ == "__main__":
    print("=== MotivaScan — Reconocimiento Facial ===")
    print("Presiona ESC para detener.")
    resultado = iniciar_reconocimiento()
    if resultado.get("nombre"):
        print("\nUltimo resultado:")
        print(f"   Nombre:     {resultado['nombre']}")
        print(f"   Emocion:    {resultado['emocion']}")
        print(f"   Sueno:      {resultado['sueno']}%")
        print(f"   Motivacion: {resultado['motivacion']}%")
        print(f"   Sonrisa:    {resultado.get('sonrisa', False)}")
        print(f"   Tension:    {resultado.get('tension', 0)}")
