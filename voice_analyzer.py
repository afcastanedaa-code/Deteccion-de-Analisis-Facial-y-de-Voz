"""
voice_analyzer.py - Reconocimiento y análisis de voz mejorado
MEJORAS:
  - Análisis de tono (frecuencia fundamental F0) con librosa si está disponible
  - Análisis de velocidad de habla (palabras por segundo)
  - Análisis de pausas y silencios
  - Análisis de energía / volumen (RMS)
  - Léxico ampliado con pesos graduados (no binario)
  - Combinación de señales acústicas + léxicas para mayor precisión
"""

import numpy as np

# ─── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_DISPONIBLE = True
except ImportError:
    SR_DISPONIBLE = False

try:
    import sounddevice as sd
    SD_DISPONIBLE = True
except Exception:
    SD_DISPONIBLE = False

try:
    import librosa
    LIBROSA_DISPONIBLE = True
except ImportError:
    LIBROSA_DISPONIBLE = False

# ─── Léxico ampliado con pesos (+2 = muy positivo, -2 = muy negativo) ─────────
LEXICO_VOZ = {
    # Muy positivo (+2)
    "excelente": 2, "genial": 2, "increible": 2, "perfecto": 2,
    "entendido": 2, "aprendí": 2, "aprendido": 2, "listo": 2,
    "claro": 2, "me encanta": 2, "me gusta": 2, "emocionado": 2,
    "emocionada": 2, "interesante": 2, "bien": 1, "entiendo": 2,
    "comprendo": 2, "puedo": 1, "seguro": 1, "segura": 1,
    "motivado": 2, "motivada": 2, "feliz": 2, "contento": 2, "contenta": 2,

    # Positivo (+1)
    "sí": 1, "si": 1, "ok": 1, "bueno": 1, "buena": 1,
    "fácil": 1, "facil": 1, "normal": 0, "regular": 0,

    # Negativo (-1)
    "no sé": -1, "no se": -1, "creo": -1, "quizás": -1, "quizas": -1,
    "tal vez": -1, "puede ser": -1, "complicado": -1, "complicada": -1,
    "difícil": -1, "dificil": -1, "no entiendo": -2,

    # Muy negativo (-2)
    "no puedo": -2, "cansado": -2, "cansada": -2, "aburrido": -2,
    "aburrida": -2, "horrible": -2, "mal": -2, "terrible": -2,
    "no quiero": -2, "perdido": -2, "perdida": -2, "confundido": -2,
    "confundida": -2, "sueño": -2, "dormido": -2, "dormida": -2,
    "estresado": -2, "estresada": -2, "ansiedad": -2, "triste": -2,
    "frustrado": -2, "frustrada": -2, "no entiendo nada": -3,
}


def analizar_lexico(texto: str) -> dict:
    """
    Análisis léxico con pesos graduados.
    Retorna score numérico y palabras encontradas.
    """
    texto_lower = texto.lower()
    score = 0
    palabras_encontradas = []

    # Primero frases (más específicas)
    for frase, peso in sorted(LEXICO_VOZ.items(), key=lambda x: -len(x[0])):
        if frase in texto_lower:
            score += peso
            tipo = "positiva" if peso > 0 else "negativa" if peso < 0 else "neutra"
            palabras_encontradas.append((tipo, frase, peso))

    return {"score": score, "palabras": palabras_encontradas}


def analizar_acustica(audio_array: np.ndarray, samplerate: int) -> dict:
    """
    Analiza características acústicas del audio:
      - Energía RMS (volumen promedio)
      - Varianza de energía (dinámica vocal)
      - F0 (tono fundamental) si librosa disponible
      - Velocidad estimada: razón de frames con voz vs total
    """
    resultados = {
        "energia_rms": 0.0,
        "varianza_energia": 0.0,
        "tono_promedio_hz": 0.0,
        "tono_varianza": 0.0,
        "ratio_voz": 0.0,
        "interpretacion": "desconocido"
    }

    if audio_array is None or len(audio_array) == 0:
        return resultados

    # Normalizar a float32
    audio_f = audio_array.astype(np.float32)
    if audio_f.max() > 1.0:
        audio_f = audio_f / 32768.0

    # Energía en ventanas de 0.05s
    win = max(1, int(samplerate * 0.05))
    frames = [audio_f[i:i+win] for i in range(0, len(audio_f) - win, win)]
    if not frames:
        return resultados

    energias = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    rms_promedio   = float(np.mean(energias))
    rms_varianza   = float(np.std(energias))

    umbral_voz = rms_promedio * 0.3
    frames_con_voz = np.sum(energias > umbral_voz)
    ratio_voz = float(frames_con_voz / len(energias)) if len(energias) > 0 else 0.0

    resultados["energia_rms"]      = round(rms_promedio, 5)
    resultados["varianza_energia"] = round(rms_varianza, 5)
    resultados["ratio_voz"]        = round(ratio_voz, 3)

    # Tono fundamental con librosa (opcional pero muy útil)
    if LIBROSA_DISPONIBLE:
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio_f,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=samplerate
            )
            f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
            if len(f0_voiced) > 0:
                resultados["tono_promedio_hz"] = round(float(np.nanmean(f0_voiced)), 1)
                resultados["tono_varianza"]    = round(float(np.nanstd(f0_voiced)), 1)
        except Exception:
            pass

    # Interpretación acústica
    # Voz con alta energía + alta varianza = expresivo/motivado
    # Voz con baja energía + poco ratio_voz = apagado/cansado/triste
    if rms_promedio < 0.01 or ratio_voz < 0.15:
        interp = "apagado"       # Muy poco audio → cansado o callado
    elif rms_promedio > 0.08 and rms_varianza > 0.04:
        interp = "expresivo"     # Mucha energía y dinámica → motivado/feliz
    elif rms_promedio > 0.05 and rms_varianza < 0.02:
        interp = "plano_alto"    # Energía alta pero monótono → estrés/tensión
    elif rms_varianza > 0.035:
        interp = "variable"      # Dinámica media → puede ser emocional
    elif ratio_voz < 0.4:
        interp = "pausado"       # Habla poco → duda/distraído
    else:
        interp = "normal"

    # Ajuste por tono si disponible
    tono = resultados["tono_promedio_hz"]
    if tono > 0:
        if tono > 220 and resultados["tono_varianza"] > 30:
            interp = "expresivo"   # Tono alto y variable = emocionado
        elif tono < 130 and rms_promedio < 0.04:
            interp = "apagado"     # Tono bajo y poco volumen = triste/cansado
        elif resultados["tono_varianza"] < 15 and tono > 150:
            interp = "plano_alto"  # Monótono con tono medio-alto = estrés

    resultados["interpretacion"] = interp
    return resultados


def combinar_analisis(score_lexico: int, acustica: dict, texto: str) -> dict:
    """
    Fusiona análisis léxico y acústico en un único resultado de motivación.
    """
    interp = acustica.get("interpretacion", "normal")
    ratio_voz = acustica.get("ratio_voz", 0.5)

    # Ajuste acústico al score léxico
    ajuste_acustico = 0
    if interp == "expresivo":
        ajuste_acustico = +3
    elif interp == "plano_alto":
        ajuste_acustico = -1    # posible estrés
    elif interp == "apagado":
        ajuste_acustico = -3
    elif interp == "pausado":
        ajuste_acustico = -2
    elif interp == "variable":
        ajuste_acustico = +1

    score_final = score_lexico + ajuste_acustico

    # Número de palabras como señal de participación
    num_palabras = len(texto.split())
    if num_palabras > 15:
        score_final += 1      # Habla mucho → participativo
    elif num_palabras < 4:
        score_final -= 1      # Respuesta muy corta → poco comprometido

    # Mapeo score → sentimiento
    if score_final >= 3:
        sentimiento = "Motivado"
        nivel = min(100, 70 + score_final * 4)
    elif score_final >= 1:
        sentimiento = "Feliz"
        nivel = min(100, 62 + score_final * 4)
    elif score_final == 0:
        sentimiento = "Neutral"
        nivel = 50
    elif score_final >= -2:
        sentimiento = "Distraido"
        nivel = max(0, 45 + score_final * 5)
    elif score_final >= -4:
        sentimiento = "Estresado" if interp == "plano_alto" else "Triste"
        nivel = max(0, 35 + score_final * 4)
    else:
        sentimiento = "Cansado"
        nivel = max(0, 20 + score_final * 3)

    return {
        "sentimiento":       sentimiento,
        "nivel_motivacion":  int(nivel),
        "score":             score_final,
        "score_lexico":      score_lexico,
        "ajuste_acustico":   ajuste_acustico,
        "interpretacion_voz": interp,
        "ratio_voz":         round(ratio_voz, 2),
        "tono_hz":           acustica.get("tono_promedio_hz", 0),
    }


def analizar_texto_motivacion(texto: str) -> dict:
    """
    Analiza únicamente el texto (sin audio).
    Compatible con la interfaz anterior.
    """
    lexico = analizar_lexico(texto)
    acustica_vacia = {"interpretacion": "normal", "ratio_voz": 0.5,
                      "tono_promedio_hz": 0, "energia_rms": 0}
    resultado = combinar_analisis(lexico["score"], acustica_vacia, texto)
    resultado["texto"] = texto
    resultado["palabras_clave"] = lexico["palabras"]
    return resultado


def escuchar_microfono(duracion: int = 5, idioma: str = "es-CO") -> dict:
    """
    Escucha el micrófono, transcribe y analiza acústica + léxico.
    """
    if not SR_DISPONIBLE:
        return {
            "exito": False,
            "error": "Librería 'speech_recognition' no instalada.\n"
                     "Ejecuta: pip install SpeechRecognition"
        }

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8   # tolerar pausas de hasta 0.8s

    PYAUDIO_DISP = False
    audio_backend_used = None
    chosen_device_name = None
    audio_array_raw = None
    samplerate_used = 44100

    try:
        import pyaudio
        with sr.Microphone() as _test:
            PYAUDIO_DISP = True
    except Exception:
        PYAUDIO_DISP = False

    audio_sr = None  # objeto AudioData de SpeechRecognition

    try:
        # ── Backend PyAudio ───────────────────────────────────────────────────
        if PYAUDIO_DISP:
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.4)
                    audio_sr = recognizer.listen(
                        source, timeout=duracion, phrase_time_limit=duracion
                    )
                    audio_backend_used = "pyaudio"
                    # Guardar array para análisis acústico
                    audio_array_raw = np.frombuffer(audio_sr.frame_data, dtype=np.int16)
                    samplerate_used  = audio_sr.sample_rate
            except Exception as e_pa:
                PYAUDIO_DISP = False
                if not SD_DISPONIBLE:
                    return {"exito": False,
                            "error": f"PyAudio falló y sounddevice no disponible: {e_pa}",
                            "backend": None, "device": None}

        # ── Backend sounddevice ───────────────────────────────────────────────
        if not PYAUDIO_DISP and SD_DISPONIBLE:
            chosen_device = None
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    try:
                        fs = int(d.get("default_samplerate", 44100))
                        sd.rec(int(0.3 * fs), samplerate=fs, channels=1,
                               dtype="int16", device=i)
                        sd.wait()
                        chosen_device = i
                        samplerate_used = fs
                        chosen_device_name = d.get("name")
                        break
                    except Exception:
                        continue

            fs = samplerate_used
            ch = 1
            frames = sd.rec(int(duracion * fs), samplerate=fs,
                            channels=ch, dtype="int16", device=chosen_device)
            sd.wait()

            if isinstance(frames, np.ndarray) and frames.ndim > 1:
                frames = frames[:, 0]
            audio_array_raw = frames
            audio_sr = sr.AudioData(frames.tobytes(), fs, 2)
            audio_backend_used = "sounddevice"

        if audio_sr is None:
            return {"exito": False, "error": "No se pudo capturar audio.",
                    "backend": None, "device": None}

        # ── Transcripción ─────────────────────────────────────────────────────
        texto = recognizer.recognize_google(audio_sr, language=idioma)

        # ── Análisis acústico ─────────────────────────────────────────────────
        acustica = analizar_acustica(audio_array_raw, samplerate_used)

        # ── Análisis léxico ───────────────────────────────────────────────────
        lexico = analizar_lexico(texto)

        # ── Fusión ────────────────────────────────────────────────────────────
        analisis = combinar_analisis(lexico["score"], acustica, texto)
        analisis["texto"] = texto
        analisis["palabras_clave"] = lexico["palabras"]
        analisis["acustica"] = acustica

        return {
            "exito":   True,
            "texto":   texto,
            "analisis": analisis,
            "backend": audio_backend_used,
            "device":  chosen_device_name,
        }

    except sr.WaitTimeoutError:
        return {"exito": False, "error": "No se detectó voz en el tiempo indicado.",
                "backend": audio_backend_used, "device": chosen_device_name}
    except sr.UnknownValueError:
        return {"exito": False, "error": "No se pudo entender el audio.",
                "backend": audio_backend_used, "device": chosen_device_name}
    except sr.RequestError as e:
        return {"exito": False, "error": f"Error del servicio de reconocimiento: {e}",
                "backend": audio_backend_used, "device": chosen_device_name}
    except OSError as e:
        return {"exito": False, "error": f"Error del dispositivo de audio: {e}",
                "backend": audio_backend_used, "device": chosen_device_name}
    except Exception as e:
        return {"exito": False, "error": f"Error inesperado: {e}",
                "backend": audio_backend_used, "device": chosen_device_name}


def verificar_dependencias() -> dict:
    estado = {"speech_recognition": SR_DISPONIBLE, "sounddevice": SD_DISPONIBLE,
              "librosa": LIBROSA_DISPONIBLE}
    try:
        import pyaudio
        estado["pyaudio"] = True
    except Exception:
        estado["pyaudio"] = False
    return estado


if __name__ == "__main__":
    print("=== Análisis de Voz Mejorado ===")
    deps = verificar_dependencias()
    for k, v in deps.items():
        print(f"  {k}: {'OK' if v else 'no instalado'}")

    audio_ok = deps.get("pyaudio") or deps.get("sounddevice")
    if deps["speech_recognition"] and audio_ok:
        print("\nHabla ahora (5 segundos)...")
        res = escuchar_microfono(5)
        if res["exito"]:
            a = res["analisis"]
            print(f"Texto:          {res['texto']}")
            print(f"Sentimiento:    {a['sentimiento']}")
            print(f"Motivacion:     {a['nivel_motivacion']}%")
            print(f"Score lexico:   {a['score_lexico']}")
            print(f"Ajuste acustico:{a['ajuste_acustico']}")
            print(f"Voz:            {a['interpretacion_voz']}")
            if a.get("tono_hz"):
                print(f"Tono promedio:  {a['tono_hz']} Hz")
        else:
            print(f"Error: {res['error']}")
    else:
        print("\nInstala: pip install SpeechRecognition sounddevice numpy")
        if not LIBROSA_DISPONIBLE:
            print("Para análisis de tono: pip install librosa")
