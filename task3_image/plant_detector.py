from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "neghani/plantvillage-mobilenetv2"
BASE_DIR = Path(__file__).resolve().parent

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

def detect_plant_disease(image_path: str) -> dict:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Изображение не найдено: {image_path}"
        )

    with Image.open(path) as source_image:
        image = source_image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3
    )

    predictions = []

    for score, index in zip(top_probabilities, top_indices):

        class_name = model.config.id2label[index.item()]

        parts = class_name.split("___", 1)

        plant_name = parts[0]
        disease_name = parts[1] if len(parts) > 1 else "Unknown"

        predictions.append({
            "plant": plant_name,
            "disease": disease_name,
            "confidence": round(score.item(), 4)
        })

    return {
        "file": path.name,
        "predictions": predictions
    }


if __name__ == "__main__":

    image_file = BASE_DIR / "samples" / "tomato_leaf.jpg"

    result = detect_plant_disease(str(image_file))

    print("\nАнализ изображения:", result["file"])
    print("Три наиболее вероятных результата:\n")

    for number, prediction in enumerate(
        result["predictions"],
        start=1
    ):
        print(f"{number}. Растение: {prediction['plant']}")
        print(f"   Состояние: {prediction['disease']}")
        print(
            f"   Уверенность: "
            f"{prediction['confidence'] * 100:.2f}%"
        )
        print()

    best_prediction = result["predictions"][0]
