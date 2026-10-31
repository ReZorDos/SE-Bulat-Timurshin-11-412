from pathlib import Path

import cv2
from ultralytics import solutions


CURRENT_DIR = Path(__file__).resolve().parent


def count_cars(video_path: str, output_path: str) -> dict:
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Файл {video_path} не найден!")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Некорректные размеры видео")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Не удалось сохранить видео")

    counter = solutions.ObjectCounter(
        model="yolo11n.pt",
        region=[(0, int(height * 0.6)),
                (width, int(height * 0.6))],
        classes=[2],
        tracker="bytetrack.yaml",
        show=False
    )

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            results = counter(frame)
            writer.write(results.plot_im)

    finally:
        cap.release()
        writer.release()

    return {
        "total_cars": counter.in_count + counter.out_count,
        "output_file": output_path
    }


if __name__ == "__main__":
    video = CURRENT_DIR / "samples" / "traffic.mp4"
    output = CURRENT_DIR / "results" / "traffic_result.mp4"

    result = count_cars(str(video), str(output))

    print("Автомобилей проехало:", result["total_cars"])
    print("Результат сохранён:", result["output_file"])
