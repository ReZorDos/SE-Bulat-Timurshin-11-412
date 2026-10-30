import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

modelName = "super-apple/spam-classifier-ru"

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSequenceClassification.from_pretrained(modelName)
model.eval()

def detect_spam(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(
        probabilities, dim=-1
    ).item()

    confidence = probabilities[0][predicted_class].item()

    return {
        "is_spam": predicted_class == 1,
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":

    text = (
        "Уважаемый получатель! Поздравляем Вас! Вы стали ПОБЕДИТЕЛЕМ нашей лотереи и выиграли 1 000 000 долларов США!!! "
        "Чтобы получить приз, НЕМЕДЛЕННО отправьте нам свои полные банковские реквизиты, номер карты, CVV-код, паспортные "
        "данные и пароль от личного кабинета!!!"
    )

    result = detect_spam(text)

    print("Сообщение:", text)
    if result["is_spam"]:
        print("Результат: СПАМ")
    else:
        print("Результат: НЕ СПАМ")
    print(
        f"Уверенность модели: "
        f"{result['confidence'] * 100:.2f}%"
    )
