import cv2
import csv
import numpy as np
from ultralytics import YOLO

CSV_PATH = "slots.csv"
MODEL_PATH = "best.pt"

CONFIDENCE = 0.25
IMGSZ = 1280

def load_slots(csv_path):
    slots = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                pts = list(map(int, row[1:9]))
                pts_tuples = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
                slots.append(pts_tuples)
        return slots
    except FileNotFoundError:
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return []

def is_car_center_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center_x = int((x1 + x2) / 2)
    car_center_y = int((y1 + y2) / 2)
    car_center = (car_center_x, car_center_y)
    
    slot_cnt = np.array(slot_points, dtype=np.int32)
    result = cv2.pointPolygonTest(slot_cnt, car_center, False)
    return result >= 0

def analyze_parking_video(video_path):
    print(f"AI 분석 시작: {video_path}")
    
    slots = load_slots(CSV_PATH)
    if not slots:
        print("슬롯 정보가 없습니다.")
        return {}

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return {}

    final_status = {}

    total_car_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, classes=[0], verbose=False)
        
        car_boxes = []
        current_car_count = 0

        for r in results:
            boxes_list = r.boxes.xyxy.tolist()
            current_car_count += len(boxes_list)

            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                car_boxes.append((x1, y1, x2, y2))

        # 현재 프레임의 슬롯 점유 상태 확인
        for idx, slot in enumerate(slots):
            slot_id = idx + 1 # 슬롯 ID는 1부터 시작
            is_occupied = False
            
            for car in car_boxes:
                if is_car_center_in_slot(car, slot):
                    is_occupied = True
                    break
            
            final_status[slot_id] = is_occupied
        
        total_car_count += current_car_count

    cap.release()
    print("분석 종료")
    
    avg_car_count = 0
    if frame_count > 0:
        avg_car_count = int(total_car_count / frame_count)
    
    vehicle_counts = {
        "car": avg_car_count
    }

    return {
        "spaces": final_status,
        "vehicles": vehicle_counts,
        "slots": {idx+1: slot for idx, slot in enumerate(slots)}
    }