from pathlib import Path
from math import gcd
import csv

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from scipy.signal import resample_poly


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent

model = hub.load(MODEL_URL)

class_names = []

with tf.io.gfile.GFile(model.class_map_path().numpy()) as file:
    reader = csv.DictReader(file)

    for row in reader:
        class_names.append(row["display_name"])


def detect_sound(file_path: str) -> dict:
    sample_rate, audio = wavfile.read(file_path)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)

        if info.min < 0:
            audio = audio.astype(np.float32) / max(
                abs(info.min), info.max
            )
        else:
            midpoint = (info.max + info.min) / 2
            audio = (
                audio.astype(np.float32) - midpoint
            ) / midpoint
    else:
        audio = audio.astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)

        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor
        )

    if audio.size == 0:
        raise ValueError("Аудиофайл пустой")

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    scores, embeddings, spectrogram = model(audio)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_indices = np.argsort(mean_scores)[-3:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "sound": class_names[index],
            "score": round(float(mean_scores[index]), 4)
        })

    return {
        "file": Path(file_path).name,
        "predictions": results
    }


if __name__ == "__main__":

    audio_file = BASE_DIR / "samples" / "miaow_16k.wav"

    result = detect_sound(str(audio_file))

    print("Аудиофайл:", result["file"])

    print("Обнаруженные звуки:")
    for prediction in result["predictions"]:
        print(
            f"{prediction['sound']}: "
            f"{prediction['score'] * 100:.2f}%"
        )
