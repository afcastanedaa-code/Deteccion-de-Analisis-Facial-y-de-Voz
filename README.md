## MotivaScan — Sistema de Detección de Motivación Estudiantil

> Aplicación de reconocimiento facial y de voz para detectar el estado emocional,
> nivel de sueño y motivación de estudiantes universitarios en tiempo real.

---

## Estructura del Proyecto

```
StudentMotivation/
─ Aplicacion.py: App principal (Streamlit - interfaz web)
─ face_register.py: Registro de estudiantes (captura de fotos)
─ face_training.py: Entrenamiento del modelo EigenFaces
─ face_recognizer.py: Reconocimiento facial + análisis emoción/sueño
─ voice_analyzer.py: Reconocimiento y análisis de voz
─ utils.py: Funciones compartidas y configuración

models/
─ frontalface_default.xml: Clasificador Haar Cascade (incluido)
─ modeloEigenFaces.xml: Modelo entrenado (se genera al entrenar)

─ DataFaces/ - Banco de imágenes de estudiantes
─ [nombre_estudiante]/
│       └── foto_001.jpg ...
│
├── registro_sesiones.json  ← Historial de sesiones (se genera automático)
└── requirements.txt        ← Dependencias
```

---

## Instalación rápida

### 1. Instalar dependencias

``` simbolo del sistema
pip install -r requirements.txt
```

> **Nota importante:** Para el reconocimiento facial necesitas `opencv-contrib-python`
> (no `opencv-python` solo):
> ```bash
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

### 2. Para reconocimiento de voz (opcional)

### cmd

### pip install SpeechRecognition

# Windows:
### pip install pyaudio


---

## Cómo usar la aplicación

### Opción A — Interfaz Streamlit (Recomendada)

``` simbolo del sistema
streamlit run app.py
```

Se abrirá en tu navegador en `http://localhost:8501`

### Opción B — Scripts individuales por consola

``` simbolo del sistema
# 1. Registrar estudiante
python face_register.py

# 2. Entrenar el modelo
python face_training.py

# 3. Iniciar reconocimiento
python face_recognizer.py

# 4. Probar análisis de voz
python voice_analyzer.py
```

---

## Flujo de uso

```
1. REGISTRAR ESTUDIANTE
   └─ Tab "Registrar Estudiante" → Ingresa nombre → Iniciar Registro
      └─ La cámara se abre, captura 100 fotos automáticamente

2. ENTRENAR MODELO
   └─ Tab "Entrenar Modelo" → Haz clic en "Entrenar Modelo Ahora"
      └─ EigenFaces aprende los patrones de cada estudiante

3. DETECTAR MOTIVACIÓN
   └─ Tab "Detección en Vivo" → Iniciar Detección
      └─ La app reconoce al estudiante y muestra:
         ● Nombre del estudiante
         ● Estado de ánimo (Motivado / Neutral / Cansado / Estresado)
         ● % de sueño (basado en apertura de ojos)
         ● % de motivación general

4. ANÁLISIS DE VOZ (opcional)
   └─ Tab "Análisis de Voz" → Escuchar ahora
      └─ Transcribe lo que dice el estudiante y detecta motivación

5. VER HISTORIAL
   └─ Tab "Historial" → Ver gráficas y exportar CSV
```

---

## Tecnologías utilizadas

| Componente | Tecnología |

| Detección facial | OpenCV + Haar Cascade |
| Reconocimiento facial | EigenFaces (opencv-contrib) |
| Análisis de sueño | Eye Aspect Ratio (apertura de ojos) |
| Análisis de emoción | Heurística geométrica + EAR |
| Reconocimiento de voz | SpeechRecognition + Google Speech API |
| Interfaz | Streamlit |
| Almacenamiento | JSON + sistema de archivos |

---

## Indicadores detectados

| Indicador | Rango | Descripción |

| **Sueño** | 0–100% | 0% = completamente despierto, 100% = dormido |
| **Motivación** | 0–100% | Índice combinado de emoción + sueño |
| **Estado** | 5 categorías | Motivado / Neutral / Distraído / Estresado / Cansado |

---

## Solución de problemas comunes

**Error: `cv2.face` no encontrado**
``` simbolo del sistema
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

**Cámara no detectada**
``` simbolo del sistema
# Verifica que la cámara esté habilitada y no usada por otra app
# Cambia el índice en face_recognizer.py: cv2.VideoCapture(1) en vez de (0)
```

**Error de pyaudio en instalación**
``` simbolo del sistema
# Windows: descarga el wheel desde https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl
```

---

## Mejoras futuras sugeridas

- Integrar **DeepFace** para análisis de emoción más preciso (7 emociones)
- Base de datos **SQLite** para persistencia robusta
- Dashboard de análisis de clase completa (múltiples estudiantes)
- Alertas automáticas al docente cuando motivación < 30%
- Exportación de reportes en PDF
- Mejora de reconocimiento de voz
- Analitica de camara y voz para mejora de comunicacion
- Anexar el Bot de DialogFlow como asistente virutal de ChatBot
