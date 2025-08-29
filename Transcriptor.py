# Transcriptor_batch_fixed.py
# Procesa todos los audios de INPUT_DIR y genera un .txt por archivo en OUTPUT_DIR
# Requisitos: pip install openai-whisper librosa soundfile
# Version estable

import os
import time
from typing import Iterable

import numpy as np
import librosa
import torch
import whisper

# ================== CONFIGURACIÓN ==================
INPUT_DIR   = r"C:\whisper_local\AudiosH"        # Carpeta con audios de entrada
OUTPUT_DIR  = r"C:\whisper_local\TranscripcionesH"  # Carpeta donde guardar los .txt
MODEL_SIZE  = "large-v3"                           # tiny | base | small | medium | large-v3
LANG        = "es"                              
USE_GPU     = True                               # True: intenta CUDA; False: fuerza CPU
FORCE_FP16  = None                               # None=auto | True | False (solo tiene sentido en CUDA)
SKIP_EXISTING = True                             # No re-procesar si ya existe el .txt
AUDIO_EXTS: Iterable[str] = (".mp3",".wav",".m4a",".flac",".ogg",".webm")
TARGET_SR   = 16000                              # Whisper trabaja a 16 kHz
# ===================================================

def pick_device() -> str:
    if USE_GPU and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_audio_array(path: str, sr: int = TARGET_SR) -> np.ndarray:
    # Carga en float32 mono a 16kHz (sin archivos temporales)
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def transcribe_numpy(model, audio_np: np.ndarray, lang: str, fp16_flag: bool) -> str:
    # Decodificación rápida/estable
    result = model.transcribe(
        audio_np,
        language=lang,
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=fp16_flag
        # Si quieres más calidad a costa de velocidad: beam_size=5
    )
    return result.get("text", "").strip()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = pick_device()
    print(f"[INFO] Device: {device} | torch {torch.__version__}")
    if device == "cuda":
        print(f"       CUDA disponible: {torch.cuda.is_available()}; GPU: {torch.cuda.get_device_name(0)}")

    # fp16: por defecto lo activamos solo en CUDA; si FORCE_FP16 está definido, obedecemos
    if FORCE_FP16 is None:
        use_fp16 = (device == "cuda")
    else:
        use_fp16 = (FORCE_FP16 and device == "cuda")

    print(f"[INFO] Cargando modelo Whisper: {MODEL_SIZE}")
    model = whisper.load_model(MODEL_SIZE, device=device)

    # Recorre los audios
    entries = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(AUDIO_EXTS))
    if not entries:
        print(f"[WARN] No se encontraron audios en: {INPUT_DIR}")
        return

    print(f"[INFO] {len(entries)} archivo(s) de audio encontrados en {INPUT_DIR}\n")

    for fname in entries:
        in_path  = os.path.join(INPUT_DIR, fname)
        base, _  = os.path.splitext(fname)
        out_path = os.path.join(OUTPUT_DIR, base + ".txt")

        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"[SKIP] Ya existe: {out_path}")
            continue

        print(f"[PROC] {fname}")
        t0 = time.time()
        try:
            audio = load_audio_array(in_path, sr=TARGET_SR)
            text  = transcribe_numpy(model, audio, LANG, use_fp16)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            dt = time.time() - t0
            print(f"  → OK ({dt:.2f}s)  Guardado: {out_path}")
        except Exception as e:
            print(f"  → ERROR procesando {fname}: {e}")

    print("\n[FIN] Lote completado.")

if __name__ == "__main__":
    main()
