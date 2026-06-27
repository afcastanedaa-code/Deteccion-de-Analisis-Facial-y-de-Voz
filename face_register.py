"""
face_register.py - Registro de nuevos estudiantes
Captura 100 fotos del rostro y las guarda en DataFaces/<nombre>/
"""

import cv2
import os
from utils import (
    cargar_clasificadores, DATA_DIR, FACE_SIZE
)


def registrar_estudiante(nombre: str, num_fotos: int = 100,
                         callback_progreso=None) -> dict:
    """
    Registra un nuevo estudiante capturando fotos de su rostro.

    Args:
        nombre: Nombre del estudiante.
        num_fotos: Cantidad de fotos a capturar (default 100).
        callback_progreso: Función opcional llamada con (contador, frame_bgr).

    Returns:
        dict con 'exito', 'fotos_guardadas', 'ruta'.
    """
    if not nombre.strip():
        return {"exito": False, "error": "El nombre no puede estar vacío."}

    face_cascade, _, _ = cargar_clasificadores()

    # Crear carpeta del usuario
    user_path = os.path.join(DATA_DIR, nombre.strip())
    os.makedirs(user_path, exist_ok=True)

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        return {"exito": False, "error": "No se pudo acceder a la cámara."}

    cont = 0

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            # Dibujar recuadro
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 200, 100), 2)

            # Extraer y guardar rostro en escala de grises para consistencia.
            rostro = frame[y:y+h, x:x+w]
            rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
            rostro = cv2.resize(rostro, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
            ruta_img = os.path.join(user_path, f"{nombre}_{cont:03d}.jpg")
            cv2.imwrite(ruta_img, rostro)
            cont += 1

        # Overlay de progreso
        progreso = int((cont / num_fotos) * 100)
        cv2.putText(frame_display,
                    f"Registrando: {nombre} | Fotos: {cont}/{num_fotos} ({progreso}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)
        cv2.putText(frame_display, "Presiona ESC para cancelar",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Callback para Streamlit (frame como bytes)
        if callback_progreso:
            callback_progreso(cont, frame_display)

        cv2.imshow("Registro de Estudiante", frame_display)

        key = cv2.waitKey(10)
        if key == 27 or cont >= num_fotos:
            break

    webcam.release()
    cv2.destroyAllWindows()

    return {
        "exito": True,
        "fotos_guardadas": cont,
        "ruta": user_path,
    }


# ── Ejecución directa ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Registro de Estudiante ===")
    nombre = input("Ingresa el nombre del estudiante: ").strip()
    resultado = registrar_estudiante(nombre)
    if resultado["exito"]:
        print(f"✅ Registro completado: {resultado['fotos_guardadas']} fotos guardadas.")
    else:
        print(f"❌ Error: {resultado.get('error')}")
