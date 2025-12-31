from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ai_module import load_slots
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "temp_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best.pt")
slots = load_slots("slots.csv")

# [핵심 변경 1] 현재 분석 결과 공유를 위한 전역 변수 선언
# 실제 서비스 단계에서는 Redis 같은 DB나 세션별 관리가 필요하지만, 데모용으로는 전역 변수가 가장 간단합니다.
latest_analysis_result = {
    "vehicles": [{"type": "car", "count": 0}],
    "spaces": [{"id": i+1, "occupied": 0} for i in range(len(slots))]
}

@app.post("/analyze")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "current.mp4")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "영상 업로드 완료", "file": "current.mp4"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {e}")

# 오버레이 함수 (기존과 동일)
def draw_overlay(frame, car_boxes, slots):
    for idx, pts in enumerate(slots):
        pts_np = np.array(pts, dtype=np.int32)
        occupied = False
        for (x1, y1, x2, y2) in car_boxes:
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            if cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0:
                occupied = True
                break
        
        slot_color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(frame, [pts_np], True, slot_color, 2)
        cv2.putText(frame, str(idx + 1), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, slot_color, 2)

    for (x1, y1, x2, y2) in car_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    return frame

# [핵심 변경 2] 스트리밍 함수에서 분석 데이터 업데이트 및 속도 조절
@app.get("/stream")
def stream_video(speed: int = 1): # speed 쿼리 파라미터 추가 (기본 1배속)
    video_path = os.path.join(UPLOAD_FOLDER, "current.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="업로드된 영상이 없습니다.")

    cap = cv2.VideoCapture(video_path)
    
    detect_interval = 5
    # 영상 속도 조절을 위한 변수
    # speed가 2면 2프레임마다 1번 처리 (즉 2배 빠름), 3이면 3배 빠름
    skip_frames = speed 
    last_car_boxes = []
    
    def generate():
        nonlocal last_car_boxes
        global latest_analysis_result # 전역 변수 사용 선언
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # [속도 조절 로직] 
            # 현재 프레임이 skip_frames의 배수가 아니면 건너뜀 (처리 안 함 -> 빨라짐)
            if speed > 1 and frame_count % skip_frames != 0:
                continue

            if frame_count == 1 or frame_count % detect_interval == 0:
            # YOLO 감지
                results = model.predict(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)

            car_boxes = []
            for r in results:
                for box in r.boxes.xyxy.tolist():
                    x1, y1, x2, y2 = map(int, box)
                    car_boxes.append((x1, y1, x2, y2))
            
            last_car_boxes = car_boxes

            # ---------------------------------------------------------
            # [중요] 여기서 최신 데이터를 전역 변수에 업데이트합니다.
            # ---------------------------------------------------------
            spaces_status = []
            for idx, pts in enumerate(slots):
                pts_np = np.array(pts, dtype=np.int32)
                occupied = 0
                for (x1, y1, x2, y2) in car_boxes:
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    if cv2.pointPolygonTest(pts_np, (cx, cy), False) >= 0:
                        occupied = 1
                        break
                spaces_status.append({"id": idx+1, "occupied": occupied})

            latest_analysis_result = {
                "vehicles": [{"type": "car", "count": len(car_boxes)}],
                "spaces": spaces_status
            }
            # ---------------------------------------------------------

            # 시각화
            frame = draw_overlay(frame, car_boxes, slots)
            _, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# [핵심 변경 3] 결과 반환 API는 이제 단순히 저장된 최신 값을 리턴
@app.get("/parking_spaces")
def parking_spaces():
    # 더 이상 여기서 cv2.VideoCapture를 하지 않습니다.
    # 스트리밍 함수가 열심히 업데이트해 놓은 값을 그냥 가져갑니다.
    return latest_analysis_result