"""
aplicacion.py - Aplicación principal Streamlit
Sistema de Detección de Motivación Estudiantil
Universidad - Reconocimiento Facial y de Voz
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from typing import List

# ─── Helpers de UI ───────────────────────────────────────────────────────────
def cargar_imagen_cv2(file) -> np.ndarray:
    data = file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    file.seek(0)
    return img


def mezclar_imagenes_cv2(files: List, alpha: float) -> np.ndarray:
    if len(files) != 2:
        return None
    imgs = []
    for f in files:
        img = cargar_imagen_cv2(f)
        if img is None:
            return None
        imgs.append(img)
    h = min(img.shape[0] for img in imgs)
    w = min(img.shape[1] for img in imgs)
    imgs = [cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) for img in imgs]
    blended = cv2.addWeighted(imgs[0], alpha, imgs[1], 1.0 - alpha, 0)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return blended


# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MotivaScan · Dashboard",
    page_icon="C:/Users/auxco/Documents/Nueva carpeta/MotivaScan_Proyecto/Icons/MotivaScan.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── CSS Personalizado (Fusión Tecno-Minimalista Cálida) ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg-main:      #f5f0e6;       /* Fondo crema arena suave */
    --sidebar-bg:   #ffffff;       /* Sidebar blanco puro nítido */
    --card-bg:      #ffffff;       /* Tarjetas contenedoras limpias */
    --card-nested:  #faf8f4;       /* Subcontenedores de métricas */
    --border-color: #e6dfd1;       /* Bordes orgánicos muy suaves */
    --text-title:   #2b241a;       /* Marrón casi negro súper elegante */
    --text-body:    #5c5346;       /* Gris orgánico para textos comunes */
    --brand-red:    #b31919;       /* Rojo institucional de la Universidad del Tolima */
    --brand-green:  #2d6a4f;       /* Verde para estados operativos y activos */
    --radius-lg:    18px;          /* Esquinas arqueadas redondeadas premium */
    --radius-sm:    10px;
}

/* ── Estructura General ─────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg-main) !important;
    color: var(--text-body) !important;
}
.stApp { background-color: var(--bg-main); }

/* ── Sidebar Estilo Japandi Minimalista ─────────── */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    padding-top: 15px;
}
section[data-testid="stSidebar"] * { color: var(--text-body) !important; }

/* ── Tarjetas y Paneles Premium ─────────────────── */
.fleur-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(43, 36, 26, 0.02);
}
.fleur-nested-card {
    background: var(--card-nested);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 16px;
    margin-bottom: 12px;
}

/* ── Etiquetas de Texto Estilizadas ──────────────── */
.label-small {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8c8170;
    margin-bottom: 6px;
}
.main-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: var(--text-title);
    margin: 0;
    line-height: 1.1;
}

/* ── Botones Redondeados de Alta Gama ────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--text-title), #4d4030) !important;
    color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 30px !important; 
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 12px rgba(43, 36, 26, 0.1) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(43, 36, 26, 0.18) !important;
    background: #1a150f !important;
}

/* ── Contenedores de Métricas de Streamlit ───────── */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px 16px !important;
}
[data-testid="metric-container"] label {
    color: #8c8170 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-title) !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 600 !important;
    font-size: 2rem !important;
}

/* ── Menú de Navegación (Tabs) Estilizado ───────── */
.stTabs [role="tablist"] {
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 24px;
}
.stTabs [role="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #9e9381 !important;
    background-color: transparent !important;
    border: none !important;
    padding: 10px 18px !important;
    font-size: 0.92rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-title) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--text-title) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--text-title) !important; }

/* Marcos de Ventana en Arco */
.fleur-arc-frame {
    border-radius: 140px 140px 15px 15px; 
    border: 1px solid var(--border-color);
    overflow: hidden;
    background: #e8e1d3;
    display: flex;
    align-items: center;
    justify-content: center;
}
.image-placeholder {
    background: var(--card-nested);
    border: 1px dashed var(--border-color);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8c8170;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Importaciones del ecosistema del Proyecto ────────────────────────────────
from utils import (
    listar_usuarios, cargar_historial, EMOCIONES, MODEL_PATH
)
from face_training import entrenar_modelo
from voice_analyzer import (
    escuchar_microfono, verificar_dependencias, analizar_texto_motivacion
)

# Carga inicial de datos compartidos
modelo_existe = os.path.exists(MODEL_PATH)
usuarios = listar_usuarios()

# ─── Encabezado Principal Premium (Logos e Identidad) ─────────────────────────
col_logo, col_titulo, col_institucion = st.columns([1, 5, 2.5])

with col_logo:
    # Corrección: Reemplazado st.image por un componente HTML para evitar fallos de archivo faltante
    st.markdown("<h1 style='font-size: 3.5rem; margin: 0; padding-top: 5px; text-align: center;'>🎓</h1>", unsafe_allow_html=True)

with col_titulo:
    st.markdown("""
    <div style="padding-top: 2px;">
        <div class="label-small">SISTEMA DE DETECCIÓN MULTIFACTORIAL</div>
        <h1 class="main-title">MotivaScan</h1>
        <p style="margin: 4px 0 0 0; color: #8c8170; font-size: 0.88rem;">
            Plataforma de IA para el Análisis de Motivación Estudiantil — Reconocimiento Facial y de Voz
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_institucion:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e6dfd1; border-radius: 12px; padding: 14px; text-align: center; margin-top: 4px;">
        <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #b31919; letter-spacing: 0.05em;">Universidad del Tolima</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 12px 0 px 0;'>", unsafe_allow_html=True)

# ─── Sidebar Lateral ──────────────────────────────────────────────────────────
with st.sidebar:
    # Corrección: Reemplazado st.image por componente HTML nativo
    st.image("C:/Users/auxco/Documents/Nueva carpeta/MotivaScan_Proyecto/Icons/MotivaScan.png", width=200)
    st.markdown("<h3 style='text-align: center; font-family:Cormorant Garamond, serif; font-size: 1.4rem; color: var(--text-title); margin-top: 0;'>MotivaScan v2.0</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown("<div class='label-small'>Estado del sistema</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Estudiantes", len(usuarios))
    col_s2.metric("Modelo IA", "Operacional" if modelo_existe else "Pendiente")

    st.divider()

    st.markdown("<div class='label-small' style='margin-bottom:8px;'>Registrados</div>", unsafe_allow_html=True)
    if usuarios:
        for u in usuarios:
            st.markdown(
                f"<div style='padding: 6px 12px; margin-bottom: 5px; background: var(--card-nested); "
                f"border-radius: 6px; border-left: 3px solid var(--brand-red); font-size: 0.85rem; color: var(--text-title);'>"
                f"👤 {u}</div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("Ninguno registrado aún.")

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#8c8170; font-family:JetBrains Mono,monospace; line-height:1.5; text-align:center;'>"
        "deep neural network • OpenCV<br>MotivaScan Ecosystem 2026"
        "</div>",
        unsafe_allow_html=True
    )

# ─── Pestañas de Navegación del Sistema ────────────────────────────────────────
tab1, tab2, tab3, tab_integrado, tab4, tab5 = st.tabs([
    "Detección en vivo",
    "Registrar estudiante",
    "Entrenar modelo",
    "Análisis integrado",
    "Análisis de voz",
    "Historial"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · DETECCIÓN EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_lbl, col_badge = st.columns([3, 1])
    with col_lbl:
        st.markdown("""
        <div style='margin-bottom: 4px;'>
            <div class='label-small'>Módulo 01</div>
            <h2 style='font-family:Cormorant Garamond,serif; font-size:1.7rem; font-weight:600; color:var(--text-title); margin:0;'>
                Resumen de Detección
            </h2>
        </div>
        """, unsafe_allow_html=True)
    with col_badge:
        st.markdown('<div class="image-placeholder" style="height:42px; font-weight:600;">● live</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:10px 0 20px;'>", unsafe_allow_html=True)

    if not modelo_existe:
        st.warning("El modelo no ha sido entrenado. Ve a 'Entrenar modelo' primero.")
    else:
        col_ctrl, col_video, col_live_metrics = st.columns([1.1, 1.8, 1.1])
        
        with col_ctrl:
            st.markdown('<div class="fleur-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="label-small">Estado del Sistema</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(45, 106, 79, 0.1); border-left: 4px solid var(--brand-green); padding: 8px 12px; border-radius: 6px; margin-bottom: 16px;">
                <span style="color: var(--brand-green); font-size: 0.85rem; font-weight: 600;">✓ ACTIVO</span>
            </div>
            """, unsafe_allow_html=True)
            
            iniciar = st.button("Iniciar detección", use_container_width=True)
            st.markdown('<div style="margin-top: 15px; font-size:0.8rem; line-height:1.5;" class="fleur-nested-card">La transmisión de video se desplegará de forma externa. Presiona <b>ESC</b> para detener de forma segura.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_video:
            st.markdown('<div class="fleur-card" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown('<div class="label-small">Cámara y Reconocimiento de Patrones</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="fleur-arc-frame" style="height: 280px; margin: 12px 0;">
                <div style="text-align: center; padding: 20px;">
                    <span style="font-size: 2.5rem;">📸</span><br>
                    <span style="font-family: 'Cormorant Garamond', serif; font-size: 1.2rem; font-style: italic; color: var(--text-body);">OpenCV - ESC para detener.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if iniciar:
                with st.spinner("Conectando con el hardware de video..."):
                    from face_recognizer import iniciar_reconocimiento
                    resultado = iniciar_reconocimiento()
                    
                    if resultado and resultado.get("exito"):
                        st.session_state["ultimo_resultado_f1"] = resultado
                        st.rerun()

        with col_live_metrics:
            st.markdown('<div class="fleur-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="label-small">Métricas en Tiempo Real</div>', unsafe_allow_html=True)
            
            res_vivo = st.session_state.get("ultimo_resultado_f1", {})
            emocion_v = res_vivo.get("emocion", "—").split()[0] if res_vivo.get("emocion") else "—"
            sueno_v = res_vivo.get("sueno", 0)
            motivacion_v = res_vivo.get("motivacion", 0)
            
            st.markdown(f"""
            <div class="fleur-nested-card">
                <span style="font-size: 0.72rem; color: #8c8170; text-transform: uppercase;">Calidad de Detección</span>
                <h3 style="color: var(--text-title); margin: 2px 0 0 0; font-family: 'Cormorant Garamond', serif; font-size: 1.9rem;">{motivacion_v}%</h3>
            </div>
            <div class="fleur-nested-card">
                <span style="font-size: 0.72rem; color: #8c8170; text-transform: uppercase;">Motivación Facial</span>
                <h3 style="color: var(--text-title); margin: 2px 0 0 0; font-family: 'Cormorant Garamond', serif; font-size: 1.9rem;">{emocion_v}</h3>
            </div>
            <div class="fleur-nested-card">
                <span style="font-size: 0.72rem; color: #8c8170; text-transform: uppercase;">Nivel de Sueño</span>
                <h3 style="color: var(--brand-red); margin: 2px 0 0 0; font-family: 'Cormorant Garamond', serif; font-size: 1.9rem;">{sueno_v}%</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
        st.markdown('<div class="label-small">Herramienta Avanzada</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="font-family:Cormorant Garamond,serif; font-size:1.4rem; color:var(--text-title); margin:0 0 12px 0;">Fusión e Interpolación Facial</h3>', unsafe_allow_html=True)
        
        fotos = st.file_uploader(
            "Carga exactamente dos capturas de rostro (.png/.jpg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
        )
        nivel = st.slider("Grado de Transición Lineal (Alpha)", 0.0, 1.0, 0.5, 0.05)
        
        if fotos and len(fotos) == 2:
            blended = mezclar_imagenes_cv2(fotos, alpha=nivel)
            if blended is not None:
                colL, colM, colR = st.columns(3)
                colL.image(cargar_imagen_cv2(fotos[0])[:, :, ::-1], caption="Captura A", use_container_width=True)
                colM.image(cargar_imagen_cv2(fotos[1])[:, :, ::-1], caption="Captura B", use_container_width=True)
                colR.image(blended, caption="Vectores Fusionados", use_container_width=True)
                
                buffer = cv2.imencode('.png', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))[1].tobytes()
                st.download_button("Descargar resultado fusionado", buffer, file_name="fusion_facial.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · REGISTRAR ESTUDIANTE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-small">Módulo 02</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif; font-size:1.7rem; color:var(--text-title); margin:0 0 15px 0;">Matrícula y Captura Biométrica</h2>', unsafe_allow_html=True)

    nombre_nuevo = st.text_input("Nombre completo del estudiante:", "")
    num_fotos = st.slider("Volumen de la ráfaga de captura", 30, 200, 100, 10)
    st.caption("La captura es automática: el sistema toma una foto por cada rostro detectado en cada cuadro de video hasta alcanzar el total seleccionado.")

    if st.button("Iniciar registro"):
        if not nombre_nuevo.strip():
            st.error("Por favor ingresa un nombre válido.")
        else:
            with st.spinner("Activando secuencia fotográfica de cámara..."):
                from face_register import registrar_estudiante
                resultado = registrar_estudiante(nombre_nuevo.strip(), num_fotos)

            if resultado["exito"]:
                st.success(f"Estudiante '{nombre_nuevo}' registrado correctamente con {resultado['fotos_guardadas']} imágenes.")
                st.rerun()
            else:
                st.error(f"Error en el hardware: {resultado.get('error')}")

    st.divider()
    st.markdown("<div class='label-small' style='margin-bottom:12px;'>Alumnos en Base de Datos Local</div>", unsafe_allow_html=True)
    usuarios_actual = listar_usuarios()
    if usuarios_actual:
        cols = st.columns(min(len(usuarios_actual), 4))
        for i, u in enumerate(usuarios_actual):
            from utils import DATA_DIR
            fotos_count = len([f for f in os.listdir(os.path.join(DATA_DIR, u)) if f.endswith((".jpg", ".png"))])
            cols[i % 4].markdown(
                f"<div class='fleur-nested-card' style='text-align:center;'> "
                f"<div class='image-placeholder' style='height:45px; margin-bottom:8px;'>👤 perfil</div>"
                f"<div style='font-weight:500; color:var(--text-title);'>{u}</div>"
                f"<div style='color:#8c8170; font-size:0.78rem;'>{fotos_count} muestras</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · ENTRENAR MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-small">Módulo 03</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif; font-size:1.7rem; color:var(--text-title); margin:0 0 15px 0;">Optimización Algebraica del Modelo</h2>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("""
        <div class="fleur-nested-card">
            <span class="label-small">EigenFaces Core</span>
            <p style='font-size:0.85rem; line-height:1.6; color:var(--text-body); margin-top:5px;'>
                El sistema ejecuta un Análisis de Componentes Principales (PCA) reduciendo las dimensiones de los píxeles recopilados para establecer descriptores vectoriales veloces y precisos.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_left:
        usuarios_train = listar_usuarios()
        if not usuarios_train:
            st.warning("No hay estudiantes registrados en la base de datos.")
        else:
            st.info(f"Sujetos listos para compilación matemática: {', '.join(usuarios_train)}")

            if st.button("Entrenar modelo ahora"):
                log_area = st.empty()
                logs = []

                def on_log(msg):
                    logs.append(msg)
                    log_area.markdown(
                        "<div style='background:var(--card-nested); border-radius:8px; padding:12px;"
                        "font-family:JetBrains Mono,monospace; font-size:0.75rem;"
                        "border:1px solid var(--border-color); max-height:180px; overflow-y:auto;'>"
                        + "<br>".join(f"• <span style='color:var(--text-title);'>{l}</span>" for l in logs)
                        + "</div>",
                        unsafe_allow_html=True
                    )

                with st.spinner("Calculando eigenvectores..."):
                    resultado = entrenar_modelo(callback_log=on_log)

                if resultado["exito"]:
                    st.success("Entrenamiento finalizado. El archivo binario de pesos locales ha sido actualizado.")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"{resultado.get('error')}")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── PESTAÑAS RESTANTES (ANÁLISIS INTEGRADO, VOZ E HISTORIAL) ──────────────────
with tab_integrado:
    st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-small">Módulo Multifactorial 04</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif; font-size:1.7rem; color:var(--text-title); margin:0 0 15px 0;">Estudio Integrado de Conducta Estudiantil</h2>', unsafe_allow_html=True)

    if not modelo_existe:
        st.warning("No hay modelo facial entrenado para enlazar con los canales acústicos.")
    else:
        col_cfg, col_info = st.columns([2, 1])
        with col_cfg:
            col_dur, col_idioma = st.columns(2)
            with col_dur:
                duracion_analisis = st.slider("Ventana de muestreo simultáneo (Segundos)", 5, 30, 10, key="slider_int")
            with col_idioma:
                idioma_integrado = st.selectbox("Variante Lingüística de Entrada", ["es-CO (Español Colombia)", "es-ES (Español España)", "en-US (English)"], index=0, key="sb_int")

            codigo_idioma_int = idioma_integrado.split()[0]
            iniciar_integrado = st.button("Iniciar análisis integrado", key="btn_integrado")

        with col_info:
            st.markdown("""
            <div class="fleur-nested-card">
                <span class="label-small">Protocolo Técnico</span>
                <ul style="margin:5px 0 0 0; padding-left:14px; font-size:0.82rem; color:var(--text-body); line-height:1.6;">
                    <li>Asegure el aislamiento acústico inicial del aula.</li>
                    <li>Indique al estudiante que hable de forma libre al encenderse el lente.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if iniciar_integrado:
            try:
                from face_recognizer import iniciar_reconocimiento
                import concurrent.futures

                with st.spinner(f"Grabando video y audio en simultáneo ({duracion_analisis}s)... ¡Habla ahora!"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        futuro_facial = executor.submit(
                            iniciar_reconocimiento, duracion_segundos=duracion_analisis
                        )
                        futuro_voz = executor.submit(
                            escuchar_microfono, duracion_analisis, codigo_idioma_int
                        )
                        resultado_facial = futuro_facial.result()
                        resultado_voz = futuro_voz.result()

                st.success("Diagnóstico de sensores completado.")

                if not resultado_voz.get("exito"):
                    st.warning(f"Canal de voz: {resultado_voz.get('error', 'no se pudo capturar audio.')}")

                motivacion_facial = resultado_facial.get("motivacion", 0)
                motivacion_vocal = resultado_voz.get("analisis", {}).get("nivel_motivacion", 0) if resultado_voz.get("exito") else 0
                motivacion_promedio = (motivacion_facial + motivacion_vocal) / 2
                
                c_f1, c_f2, c_f3 = st.columns(3)
                c_f1.metric("Aptitud Facial", f"{motivacion_facial}%")
                c_f2.metric("Ánimo Vocálico", f"{motivacion_vocal}%")
                c_f3.metric("Fusión Promedio", f"{motivacion_promedio:.1f}%")

                color_final = "var(--brand-green)" if motivacion_promedio > 60 else "var(--brand-red)"
                texto_transcrito = resultado_voz.get("texto") if resultado_voz.get("exito") else "Sin señal acústica legible"
                st.markdown(f"""
                <div style="border-left: 4px solid {color_final}; background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid var(--border-color); margin-top:15px;">
                    <h4 style="margin: 0 0 5px 0; font-family:'Cormorant Garamond', serif; font-size:1.3rem; color: var(--text-title);">Interpretación Diagnóstica</h4>
                    <p style="margin: 0; font-size: 0.88rem; line-height:1.6;">
                        El alumno evaluado como <b>{resultado_facial.get('nombre', 'Desconocido')}</b> reflejó un estado facial predominante <b>{resultado_facial.get('emocion', 'Neutral')}</b>. 
                        En el plano lingüístico se transcribió: <i style="color:#8c8170;">"{texto_transcrito}"</i>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Fallo en los canales integrados: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-small">Módulo 05</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif; font-size:1.7rem; color:var(--text-title); margin:0 0 15px 0;">Procesamiento del Lenguaje Natural</h2>', unsafe_allow_html=True)

    deps = verificar_dependencias()
    voz_disponible = deps.get('speech_recognition', False) and (deps.get('pyaudio', False) or deps.get('sounddevice', False))

    if not voz_disponible:
        st.warning("Dispositivos de grabación locales no mapeados.")
        texto_manual = st.text_area("Procesamiento léxico alternativo por entrada de texto:")
        if st.button("Analizar semántica") and texto_manual.strip():
            resultado = analizar_texto_motivacion(texto_manual)
            st.metric("Sentimiento Determinado", resultado["sentimiento"])
    else:
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            duracion_voz = st.slider("Segundos de escucha activa del micrófono", 3, 15, 5, key="slider_v")
            idioma_voz = st.selectbox("Idioma lingüístico", ["es-CO (Español Colombia)", "es-ES (Español España)"], index=0, key="sb_v")
            if st.button("Encender captura de micrófono", use_container_width=True):
                with st.spinner("Sintonizando micrófono del hardware..."):
                    resultado = escuchar_microfono(duracion_voz, idioma_voz.split()[0])
                if resultado["exito"]:
                    st.write(f"**Texto Transcrito:** *{resultado['texto']}*")
                    st.metric("Puntaje Motivacional Vocálico", f"{resultado['analisis']['nivel_motivacion']}%")
                else:
                    st.error(f"No se pudo capturar/transcribir audio: {resultado.get('error', 'error desconocido.')}")
        with col_v2:
            st.markdown('<div class="image-placeholder" style="height:110px;">🎙️ espectrograma de onda de audio</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="fleur-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-small">Módulo 06</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Cormorant Garamond,serif; font-size:1.7rem; color:var(--text-title); margin:0 0 15px 0;">Bitácora Histórica de Evaluaciones</h2>', unsafe_allow_html=True)

    historial = cargar_historial()
    if not historial:
        st.info("No se registran datos o bitácoras previas guardadas en el disco local.")
    else:
        df = pd.DataFrame(historial)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["motivacion_pct", "sueno_pct", "nombre", "emocion"]:
            if col not in df.columns:
                df[col] = 0 if col.endswith("_pct") else "—"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sesiones Totales", len(df))
        c2.metric("Motivación Promedio", f"{df['motivacion_pct'].mean():.1f}%")
        c3.metric("Fatiga Promedio", f"{df['sueno_pct'].mean():.1f}%")
        c4.metric("Alumnos Evaluados", df["nombre"].nunique())

        st.divider()

        filtro_nombre = st.selectbox("Filtrar bitácora por estudiante:", ["Todos"] + sorted(df["nombre"].unique().tolist()))
        df_filtrado = df if filtro_nombre == "Todos" else df[df["nombre"] == filtro_nombre]

        if len(df_filtrado) > 1:
            chart_data = df_filtrado.set_index("timestamp")[["motivacion_pct", "sueno_pct"]]
            chart_data.columns = ["Motivación (%)", "Sueño (%)"]
            st.line_chart(chart_data)

        st.markdown("<div class='label-small' style='margin-top:20px; margin-bottom:10px;'>Distribución de Estados Emocionales</div>", unsafe_allow_html=True)
        emocion_counts = df_filtrado["emocion"].value_counts()

        col_chart, col_list = st.columns([2, 1])
        with col_chart:
            st.bar_chart(emocion_counts)
        with col_list:
            for emocion, count in emocion_counts.items():
                pct = count / len(df_filtrado) * 100
                color = EMOCIONES.get(emocion, {}).get("hex", "#7a6e5f")
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between;"
                    f"padding:7px 0; border-bottom:1px solid #e6dfd1;"
                    f"font-size:0.86rem; color:var(--text-title);'>"
                    f"<span>{emocion}</span>"
                    f"<b style='color:{color}'>{pct:.0f}%</b></div>",
                    unsafe_allow_html=True
                )

        st.divider()

        st.markdown("<div class='label-small' style='margin-bottom:10px;'>Registros Detallados</div>", unsafe_allow_html=True)
        df_display = df_filtrado[["timestamp", "nombre", "emocion", "sueno_pct", "motivacion_pct"]].copy()
        df_display.columns = ["Fecha / Hora", "Nombre", "Estado", "Sueño %", "Motivación %"]
        df_display = df_display.sort_values("Fecha / Hora", ascending=False)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        csv = df_display.to_csv(index=False).encode("utf-8")
        nombre_archivo = f"motivascan_reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        st.download_button("Exportar Reporte Comercial (CSV)", data=csv, file_name=nombre_archivo, mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)