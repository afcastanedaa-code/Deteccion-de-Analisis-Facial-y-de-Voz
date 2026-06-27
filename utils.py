"""
utils.py - Funciones compartidas para análisis facial
Sistema de Detección de Motivación Estudiantil
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# ─── Rutas del proyecto ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "DataFaces")
LOG_FILE = os.path.join(BASE_DIR, "registro_sesiones.json")

FACE_CASCADE_PATH = os.path.join(MODELS_DIR, "frontalface_default.xml")
EYE_CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_eye.xml"
MOUTH_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_smile.xml"
MODEL_PATH = os.path.join(MODELS_DIR, "modeloEigenFaces.xml")

# ─── Constantes de análisis ───────────────────────────────────────────────────
FACE_SIZE = (150, 150)

# Umbral EigenFaces: valores MÁS BAJOS = reconocimiento más seguro.
# Con buena iluminación y 100 fotos, el match suele estar entre 800–2500.
# Subimos el umbral de aceptación para no rechazar rostros conocidos.
MIN_CONFIDENCE = 5500

# ─── Paleta de emociones ampliada ─────────────────────────────────────────────
# 15 estados de ánimo con señales visuales y pesos de motivación base
EMOCIONES = {
    "Neutral":      {"color": (190, 190, 60), "hex": "#BEBE3C", "peso": 74},
    "Feliz":        {"color": (0, 200, 200),  "hex": "#00C8C8", "peso": 92},
    "Motivado":     {"color": (0, 210, 90),   "hex": "#00D25A", "peso": 95},
    "Entusiasmado": {"color": (255, 180, 70), "hex": "#FFB446", "peso": 90},
    "Confiado":     {"color": (70, 190, 120), "hex": "#47BE78", "peso": 85},
    "Relajado":     {"color": (100, 210, 160),"hex": "#64D0A0", "peso": 82},
    "Sorprendido":  {"color": (220, 120, 0),  "hex": "#DC7800", "peso": 78},
    "Alerta":       {"color": (210, 140, 50), "hex": "#D28C32", "peso": 60},
    "Curioso":      {"color": (140, 170, 210),"hex": "#8CAAD2", "peso": 72},
    "Pensativo":    {"color": (170, 120, 170),"hex": "#AA79AA", "peso": 66},
    "Confundido":   {"color": (170, 90, 140), "hex": "#A75A8C", "peso": 62},
    "Distraido":    {"color": (200, 90, 40),  "hex": "#C85A28", "peso": 50},
    "Ansioso":      {"color": (220, 80, 120), "hex": "#DC5078", "peso": 40},
    "Miedo":        {"color": (180, 60, 140), "hex": "#B43C8C", "peso": 34},
    "Triste":       {"color": (120, 80, 200), "hex": "#7850C8", "peso": 28},
    "Estresado":    {"color": (50, 50, 240),  "hex": "#3232F0", "peso": 24},
    "Cansado":      {"color": (40, 130, 255), "hex": "#2882FF", "peso": 18},
    "Decepcionado": {"color": (120, 60, 120), "hex": "#784078", "peso": 22},
    "Despreocupado": {"color": (100, 175, 140),"hex":"#64AF8C", "peso": 70},
}

# ─── Cargar clasificadores ────────────────────────────────────────────────────
def cargar_clasificadores():
    face_cascade  = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade   = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(f"No se encontró clasificador facial: {FACE_CASCADE_PATH}")
    return face_cascade, eye_cascade, mouth_cascade


def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        recognizer = cv2.face.EigenFaceRecognizer_create()
    except AttributeError as e:
        raise RuntimeError(
            "OpenCV no tiene el módulo face. Instala opencv-contrib-python en vez de opencv-python."
        ) from e
    recognizer.read(MODEL_PATH)
    return recognizer


def listar_usuarios():
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]


# ─── Preprocesamiento mejorado ────────────────────────────────────────────────
def preprocesar_rostro(img_gray):
    """
    Normalización robusta: CLAHE + redimensión bicúbica + normalización de intensidad.
    Es CRÍTICO que el preprocesamiento en entrenamiento y en reconocimiento sea idéntico.
    """
    # Ecualización adaptativa (mucho mejor que equalizeHist global)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_gray)
    # Redimensión con interpolación bicúbica
    img = cv2.resize(img, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    # Normalización de intensidad 0-255 para igualar condiciones de luz
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img


# ─── Análisis de sueño (Eye Aspect Ratio mejorado) ───────────────────────────
def calcular_porcentaje_sueno(gray_frame, rostro_rect, eye_cascade):
    """
    EAR mejorado: busca ojos solo en la mitad superior del rostro para
    evitar falsas detecciones de boca/mejillas como "ojos".
    """
    x, y, w, h = rostro_rect
    # Solo mitad superior del rostro para los ojos
    roi_ojos = gray_frame[y : y + int(h * 0.6), x : x + w]
    ojos = eye_cascade.detectMultiScale(
        roi_ojos, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20)
    )

    if len(ojos) == 0:
        return 80  # Sin ojos detectados → probablemente cansado

    # Tomar los 2 ojos más grandes (filtrar ruido)
    ojos_sorted = sorted(ojos, key=lambda e: e[2] * e[3], reverse=True)[:2]
    aperturas = [eh / (h * 0.6) for (_, _, _, eh) in ojos_sorted]
    promedio = np.mean(aperturas)

    # apertura_normal ≈ 0.18, cerrado ≈ 0.05
    sueno = max(0, min(100, int((1 - (promedio - 0.05) / 0.14) * 100)))
    return sueno


# ─── Detección geométrica de sonrisa ─────────────────────────────────────────
def detectar_sonrisa(gray_frame, rostro_rect, mouth_cascade):
    """
    Retorna True si hay sonrisa detectada en la zona de la boca.
    """
    x, y, w, h = rostro_rect
    roi_boca = gray_frame[y + int(h * 0.55) : y + h, x : x + w]
    sonrisas = mouth_cascade.detectMultiScale(
        roi_boca, scaleFactor=1.7, minNeighbors=20, minSize=(25, 15)
    )
    return len(sonrisas) > 0


def detectar_boca_abierta(gray_frame, rostro_rect):
    """Detecta apertura de boca en base a la proporción de la región inferior del rostro."""
    x, y, w, h = rostro_rect
    roi_boca = gray_frame[y + int(h * 0.56) : y + h, x + int(w * 0.14) : x + int(w * 0.86)]
    if roi_boca.size == 0:
        return False
    _, umbral = cv2.threshold(roi_boca, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contornos:
        _, _, _, alto = cv2.boundingRect(cnt)
        if alto > roi_boca.shape[0] * 0.35:
            return True
    return False


def evaluar_cejas(gray_frame, rostro_rect):
    """Devuelve una intensidad de cejas basada en gradientes en la zona de frente/cejas."""
    x, y, w, h = rostro_rect
    roi_cejas = gray_frame[y + int(h * 0.08) : y + int(h * 0.25), x + int(w * 0.12) : x + int(w * 0.88)]
    if roi_cejas.size == 0:
        return 0
    grad_x = cv2.Sobel(roi_cejas, cv2.CV_64F, 1, 0, ksize=3)
    intensidad = np.mean(np.abs(grad_x))
    return min(100, int(intensidad * 4))


def evaluar_reflejos_nariz(gray_frame, rostro_rect):
    """Estima el nivel de actividad / reflejo en la zona de la nariz."""
    x, y, w, h = rostro_rect
    roi_nariz = gray_frame[y + int(h * 0.26) : y + int(h * 0.52), x + int(w * 0.35) : x + int(w * 0.65)]
    if roi_nariz.size == 0:
        return 0
    lap = cv2.Laplacian(roi_nariz, cv2.CV_64F)
    reflejo = np.mean(np.abs(lap))
    return min(100, int(reflejo * 1.8))


# ─── Análisis de tensión facial (frente / cejas) ─────────────────────────────
def calcular_tension_facial(gray_frame, rostro_rect):
    """
    Estima tensión usando gradientes en la zona de cejas/frente.
    Alta varianza de gradiente → expresión tensa/estresada.
    Retorna valor 0-100.
    """
    x, y, w, h = rostro_rect
    # Zona de cejas: franja del 20% al 45% de altura del rostro
    zona = gray_frame[y + int(h * 0.20) : y + int(h * 0.45), x : x + w]
    if zona.size == 0:
        return 50
    grad_x = cv2.Sobel(zona, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(zona, cv2.CV_64F, 0, 1, ksize=3)
    magnitud = np.sqrt(grad_x**2 + grad_y**2)
    tension = min(100, int(np.std(magnitud) / 3))
    return tension


# ─── Motor de emociones ampliado ──────────────────────────────────────────────
def analizar_emocion(porcentaje_sueno, confianza_reconocimiento,
                     hay_sonrisa=False, tension_facial=50,
                     boca_abierta=False, cejas_intensas=0,
                     reflejos_nariz=0):
    """
    Determina estado emocional usando un patrón de rostro completo:
      - ojos / sueño
      - sonrisa / boca abierta
      - tensión y cejas
      - nariz / reflejos
      - confianza del modelo

    El valor predeterminado es Neutral y solo cambia hacia emociones expresivas
    cuando hay señales faciales claras.
    """
    ojos_abiertos = porcentaje_sueno < 30
    ojos_cerrados = porcentaje_sueno > 60
    sonrisa = hay_sonrisa
    cejas_activas = cejas_intensas > 35
    nariz_activa = reflejos_nariz > 22
    tension_alta = tension_facial > 70
    cansancio = ojos_cerrados and not sonrisa
    sorpresa = ojos_abiertos and boca_abierta and cejas_activas
    miedo = ojos_abiertos and boca_abierta and cejas_activas and tension_facial > 55
    estresado = not sonrisa and cejas_activas and tension_alta
    ansiedad = not sonrisa and cejas_activas and tension_facial > 55 and nariz_activa
    felicidad = sonrisa and tension_facial < 45
    confianza = sonrisa and tension_facial >= 45
    tristeza = not sonrisa and (porcentaje_sueno > 35 or tension_facial < 40)
    pensativo = not sonrisa and ojos_abiertos and cejas_activas and tension_facial < 55
    distraido = not sonrisa and ojos_abiertos and tension_facial < 45 and porcentaje_sueno > 30

    # Nota: "Estresado" se evalúa antes que "Ansioso" porque su condición
    # (tensión > 70) es un subconjunto más estricto que la de ansiedad
    # (tensión > 55). Si se evaluara después, "Ansioso" siempre la interceptaría
    # primero y "Estresado" nunca se alcanzaría.
    if cansancio:
        return "Cansado"
    if miedo:
        return "Miedo"
    if sorpresa:
        return "Sorprendido"
    if estresado:
        return "Estresado"
    if ansiedad:
        return "Ansioso"
    if felicidad:
        return "Feliz"
    if confianza and not sonrisa:
        return "Confiado"
    if sonrisa:
        return "Entusiasmado"
    if tristeza:
        return "Triste"
    if pensativo:
        return "Pensativo"
    if distraido:
        return "Distraido"

    return "Neutral"


def obtener_emociones_candidatas(porcentaje_sueno, confianza_reconocimiento,
                                 hay_sonrisa=False, tension_facial=50,
                                 boca_abierta=False, cejas_intensas=0,
                                 reflejos_nariz=0):
    """Devuelve hasta tres emociones probables para expresiones faciales similares."""
    candidatos = []
    ojos_cerrados = porcentaje_sueno > 60
    sonrisa = hay_sonrisa
    for emocion, meta in EMOCIONES.items():
        score = meta["peso"]
        if ojos_cerrados:
            if emocion == "Cansado":
                score += 32
            elif emocion in {"Triste", "Ansioso", "Estresado"}:
                score += 18
            else:
                score -= 10
        if sonrisa:
            if emocion in {"Feliz", "Entusiasmado", "Confiado", "Motivado"}:
                score += 28
            else:
                score -= 8
        if boca_abierta:
            if emocion in {"Sorprendido", "Miedo", "Ansioso"}:
                score += 22
            else:
                score -= 6
        if cejas_intensas > 35:
            if emocion in {"Ansioso", "Estresado", "Pensativo", "Sorprendido"}:
                score += 20
            else:
                score -= 4
        if reflejos_nariz > 20:
            if emocion in {"Ansioso", "Miedo", "Estresado"}:
                score += 16
            else:
                score -= 6
        if confianza_reconocimiento < 1800 and emocion in {"Confiado", "Motivado", "Relajado"}:
            score += 12
        if confianza_reconocimiento >= 3600 and emocion in {"Sorprendido", "Triste", "Estresado"}:
            score += 14
        candidatos.append((emocion, score))

    candidatos.sort(key=lambda item: item[1], reverse=True)
    return [emocion for emocion, _ in candidatos[:3]]


def calcular_motivacion(emocion, sueno, tension_facial=50):
    """Calcula índice de motivación 0-100 usando los pesos de EMOCIONES."""
    base = EMOCIONES.get(emocion, {}).get("peso", 50)
    penalizacion_sueno = sueno * 0.30
    penalizacion_tension = max(0, (tension_facial - 45) * 0.18)
    return max(0, min(100, int(base - penalizacion_sueno - penalizacion_tension)))


# ─── Guardar / cargar registros ───────────────────────────────────────────────
def guardar_registro(nombre, emocion, sueno, motivacion):
    registros = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                registros = json.load(f)
            except Exception:
                registros = []
    registros.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nombre":        nombre,
        "emocion":       emocion,
        "sueno_pct":     sueno,
        "motivacion_pct": motivacion,
    })
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)


def cargar_historial():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []
