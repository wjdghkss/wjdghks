import cv2
import os
# DeepFace를 사용하여 YOLO 탐지 기능을 활용합니다.
from deepface import DeepFace 
import numpy as np

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
# 🚨 [필수 반영] 휴대폰 스트림 URL로 변경
STREAM_URL = "http://192.168.120.242:8080/video" 

SAVE_DIR = "detected_face"
SAVE_PATH = os.path.join(SAVE_DIR, "authorized_face.jpg")
# 🚨 [수정] main.py와 동일하게 YOLO 탐지기 사용
DETECTOR_BACKEND = "yolo" 

# 저장 디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ----------------------------------------------------
# 2. 얼굴 캡처 및 저장 함수 (YOLO 탐지 로직 적용)
# ----------------------------------------------------
def capture_and_save_face():
    print("[INFO] 얼굴 등록을 시작합니다. 카메라를 응시하고 's' 키를 누르면 얼굴이 저장됩니다.")
    print("[INFO] 종료하려면 'q'를 누르세요.")

    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다. URL({STREAM_URL}) 또는 Wi-Fi 연결을 확인하세요.")
        return

    # DeepFace 모델 로드 (YOLO 백엔드를 위해)
    try:
        print(f"[INFO] 탐지기({DETECTOR_BACKEND})를 로드 중입니다...")
        # DeepFace의 탐지 기능을 사용하기 위해 초기 로딩을 시도
        _ = DeepFace.extract_faces(img_path=np.zeros((100,100,3)), detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        print("[INFO] 탐지기 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 탐지기 로드 실패: {e}")
        cap.release()
        return

    detected_face_obj = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임을 읽지 못했습니다. 스트림 연결을 확인하세요.")
            break
        
        frame = cv2.flip(frame, 1) # 거울 모드

        # 🚨 [수정] Haar Cascade 대신 DeepFace의 YOLO 탐지기 사용
        face_objs = []
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False # 탐지 실패 시 오류 방지
            )
        except Exception:
            pass 

        # 첫 번째 탐지된 얼굴만 사용
        if len(face_objs) > 0:
            detected_face_obj = face_objs[0]
            
            # DeepFace 결과에서 좌표 추출
            x = detected_face_obj['facial_area']['x']
            y = detected_face_obj['facial_area']['y']
            w = detected_face_obj['facial_area']['w']
            h = detected_face_obj['facial_area']['h']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            detected_face_obj = None

        cv2.imshow("Authorize Face - Press 's' to save, 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 얼굴이 탐지되었을 때만 저장
            if detected_face_obj is not None:
                # YOLO가 탐지한 정확한 영역 사용
                x = detected_face_obj['facial_area']['x']
                y = detected_face_obj['facial_area']['y']
                w = detected_face_obj['facial_area']['w']
                h = detected_face_obj['facial_area']['h']
                
                # 얼굴 인식률 향상을 위해 여백(padding) 추가 로직 (기존 유지)
                pad_w = int(w * 0.25)
                pad_h = int(h * 0.25)
                
                img_h, img_w, _ = frame.shape
                new_x = max(0, x - pad_w)
                new_y = max(0, y - pad_h)
                new_w = min(img_w - new_x, w + 2 * pad_w)
                new_h = min(img_h - new_y, h + 2 * pad_h)

                padded_face_img = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                
                cv2.imwrite(SAVE_PATH, padded_face_img)
                print(f"[SUCCESS] 얼굴이 '{SAVE_PATH}'에 성공적으로 저장되었습니다.")
                break
            else:
                print("[WARNING] 저장할 얼굴이 탐지되지 않았습니다. 다시 시도해주세요.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 얼굴 등록 프로그램을 종료합니다.")

# ----------------------------------------------------
# 3. 실행
# ----------------------------------------------------
if __name__ == '__main__':
    capture_and_save_face()