import cv2
import os
from deepface import DeepFace
import numpy as np
import time

# ============================================================
# 1. 환경 설정 및 성능 제어 변수 (탐지 및 인증 최적화)
# ============================================================
STREAM_URL = "http://192.168.120.242:8080/video" 
FACE_DATABASE_DIR = "detected_face"
# 🚀 [인증 모델 최적화] 포즈 변화에 강한 ArcFace 사용 (등록된 사람의 안정성 향상)
MODEL_NAME = "ArcFace"          
# 🚨 [탐지 모델 최적화] 옆모습 탐지 강화를 위해 YOLOv5n 사용 (미등록 인물 모자이크 안정성 향상)
DETECTOR_BACKEND = "yolo"       
DISTANCE_METRIC = "cosine"      # ArcFace에 최적화된 거리 측정 방식
MOSAIC_FACTOR = 20              
PROCESS_INTERVAL = 5            

# ... (2. find_authorized_face_path 함수 동일) ...
# ... (3. apply_mosaic 함수 동일) ...

# ============================================================
# 4. 메인 실행 함수 (YOLO 탐지 로직 적용)
# ============================================================
def run_mosaic_app():
    AUTHORIZED_FACE_PATH = find_authorized_face_path(FACE_DATABASE_DIR)
    
    # ... (등록 파일 유무 확인 로직 동일) ...
    
    print(f"[INFO] 등록된 얼굴 파일을 사용합니다: {AUTHORIZED_FACE_PATH}")

    cap = cv2.VideoCapture(STREAM_URL)
    
    # ... (스트림 오픈 확인 로직 동일) ...

    # DeepFace 모델 미리 로드
    try:
        print(f"[INFO] 얼굴 인식 모델({MODEL_NAME}) 및 탐지기({DETECTOR_BACKEND})를 로드 중입니다...")
        DeepFace.build_model(MODEL_NAME)
        # ⚠️ YOLO 백엔드를 사용하려면 'ultralytics' 라이브러리가 설치되어 있어야 합니다!
        print("[INFO] 모델 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        cap.release()
        return

    print("----------------------------------------------------")
    print("[INFO] YOLO 기반 얼굴 감지를 시작합니다...")

    frame_count = 0
    is_verified = False # 직전 프레임의 인증 상태를 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 스트림에서 프레임을 읽지 못했습니다. 루프를 종료합니다.")
            break

        frame = cv2.flip(frame, 1)

        # ----------------------------------------------------
        # 🚨 YOLO를 사용하여 얼굴 탐지 (Haar Cascade 대체)
        # ----------------------------------------------------
        face_objs = []
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False # 탐지 실패 시 오류 발생 방지
            )
        except Exception:
            pass # 탐지된 얼굴이 없으면 무시

        should_process = (frame_count % PROCESS_INTERVAL == 0)
        frame_count += 1 

        # 탐지된 모든 얼굴에 대해 루프 실행
        for face_obj in face_objs:
            # DeepFace 탐지 결과에서 얼굴 이미지와 영역 좌표를 추출
            x, y, w, h = face_obj['facial_area']['x'], face_obj['facial_area']['y'], face_obj['facial_area']['w'], face_obj['facial_area']['h']
            current_face_img = face_obj['face'] 

            if current_face_img.size == 0:
                continue

            if should_process:
                # N 프레임마다 한 번씩 느린 DeepFace 연산 실행
                try:
                    result = DeepFace.verify(
                        img1_path=current_face_img,
                        img2_path=AUTHORIZED_FACE_PATH,
                        model_name=MODEL_NAME, # ArcFace 사용
                        distance_metric=DISTANCE_METRIC, # Cosine 사용
                        enforce_detection=False 
                    )
                    is_verified = result['verified']
                except Exception:
                    is_verified = False
            
            # 결과 반영: 미등록 인물(False)은 모자이크 처리
            if not is_verified:
                frame = apply_mosaic(frame, (x, y, w, h))
            else:
                # 일치하면 녹색 사각형 표시
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Authorized", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLO Optimized Mosaic - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료합니다.")

if __name__ == '__main__':
    run_mosaic_app()