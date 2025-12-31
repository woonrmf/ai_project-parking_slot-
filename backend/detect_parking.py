import cv2
import csv
import numpy as np
from ultralytics import YOLO

# -------------------------------
# 설정
VIDEO_PATH = "videos/test.mp4"       # 분석할 영상
OUTPUT_PATH = "videos/output/output_parking_(7).mp4"   # 결과 저장
MODEL_PATH = "best.pt"  # 학습된 YOLO 모델
CONFIDENCE = 0.25
IOU = 0.3
IMGSZ = 1280

# -------------------------------
# 슬롯 로드
def load_slots(csv_path="slots.csv"):
    slots = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # header skip
        for row in reader:
            pts = list(map(int, row[1:9]))
            pts = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
            slots.append(pts)
    return slots

# -------------------------------
# IoU 기준으로 슬롯 안 차량 체크
def is_car_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    sx = [p[0] for p in slot_points]
    sy = [p[1] for p in slot_points]

    # 슬롯 영역 bounding box
    bx1, by1, bx2, by2 = min(sx), min(sy), max(sx), max(sy)

    # 교차 영역 계산
    inter_x1 = max(x1, bx1)
    inter_y1 = max(y1, by1)
    inter_x2 = min(x2, bx2)
    inter_y2 = min(y2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    car_area = (x2 - x1) * (y2 - y1)
    slot_area = (bx2 - bx1) * (by2 - by1)

    # IoU 계산
    iou = inter_area / float(car_area + slot_area - inter_area + 1e-6)
    return iou > 0.2  # IoU 0.2 이상이면 점유로 간주

# -------------------------------
def main():
    slots = load_slots()
    total_slots = len(slots)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("주차 감지 시작... q 누르면 종료됨")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 차량 탐지
        results = model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, iou=IOU, verbose=False)
        car_boxes = []
        for r in results:
            for box in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, box)
                car_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 차량은 노란색

        # 슬롯별 점유 여부
        occupied_count = 0
        for idx, slot in enumerate(slots):
            occupied = any(is_car_in_slot(car, slot) for car in car_boxes)
            if occupied:
                occupied_count += 1
            color = (0, 0, 255) if occupied else (0, 255, 0)  # 빨강=점유, 초록=빈자리

            pts = np.array(slot, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, f"{idx+1}", (slot[0][0], slot[0][1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        empty_count = total_slots - occupied_count
        # 프레임 상단에 현재 점유/빈자리 표시
        cv2.putText(frame, f"Total Slots: {total_slots}  Occupied: {occupied_count}  Empty: {empty_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out_video.write(frame)
        cv2.imshow("Parking Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"총 슬롯: {total_slots}, 빈 자리: {empty_count}")
    print("출력 영상:", OUTPUT_PATH)
    print("완료")

if __name__ == "__main__":
    main()
