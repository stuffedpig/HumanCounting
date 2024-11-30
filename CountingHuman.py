'''
pip install ultralytics pandas opencv-python
install 해야 함!
'''

import cv2
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import os
import time

# CSV 파일 경로
OUTPUT_FILE = "detection_log.csv"
VIDEO_OUTPUT = "output_video.avi"  # 저장할 비디오 파일 경로

def initialize_csv():
    # CSV 파일 초기화 (파일이 없으면 헤더 추가)
    if not os.path.exists(OUTPUT_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Number of People"])
        df.to_csv(OUTPUT_FILE, index=False)

def append_to_csv(timestamp, num_people):
    # CSV 파일에 타임스탬프와 사람 수 기록
    df = pd.DataFrame([[timestamp, num_people]], columns=["Timestamp", "Number of People"])
    df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

def detect_people():
    # YOLOv8 모델 로드
    model = YOLO("yolov8n.pt")  # 'n'은 Nano 모델로 빠르고 가벼움
    initialize_csv()

    # 카메라 캡처 시작
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
    fps = 20.0  # 초당 프레임 수
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))

    print("YOLOv8 기반 사람 감지 시작. 'q'를 눌러 종료하세요.")

    last_capture_time = time.time()  # 마지막 CSV 저장 시간 초기화

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # YOLO 모델로 사람 감지
        results = model(frame)

        # 사람 클래스 필터링 (YOLO 클래스 ID: 0은 'person')
        num_people = sum(1 for result in results[0].boxes.cls if int(result) == 0)

        # 감지 결과 중 최고 정확도 계산
        confidences = results[0].boxes.conf.tolist()  # 박스들의 정확도 리스트
        min_confidence = min(confidences) if confidences else 0.0 # 최저 정확도 저장
        '''
        최저 정확도 표시는 해두는데
        confidence에 정확도 리스트 있으니 정확도가 몇 이상일 때만 counting 하거나
        최저 정확도가 몇 이상이여야만 자료에 넣거나로 하세요.
        아니면 최고 정확도로 하려면 max 써서 하면 됩니다.
        '''

        # 5초마다 CSV 파일에 기록
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_to_csv(timestamp, num_people)
            print(f"{timestamp}, 감지된 사람 수: {num_people}, 최고 정확도: {min_confidence:.2f}")
            last_capture_time = current_time

        # 감지 결과를 화면에 시각적으로 표시
        annotated_frame = results[0].plot()  # 감지된 객체를 네모 박스로 표시 -> 안원하면 주석처리

        # 오른쪽 위에 텍스트 표시 (사람 수와 정확도)
        text = f"People: {num_people}, Min Confidence: {min_confidence:.2f}"
        cv2.putText( # 안원하면 주석처리
            annotated_frame, text, (10, 30),  # 텍스트 위치 (왼쪽 상단)
            cv2.FONT_HERSHEY_SIMPLEX,  # 글꼴
            1,  # 글자 크기
            (0, 255, 0),  # 글자 색 (녹색)
            2  # 글자 굵기
        )

        # 결과를 화면에 표시
        cv2.imshow("YOLOv8 - People Detection", annotated_frame)

        # 비디오 파일에 저장
        out.write(annotated_frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"프로그램 종료. 결과는 '{OUTPUT_FILE}'에 저장되었으며, 영상은 '{VIDEO_OUTPUT}'에 저장되었습니다.")

if __name__ == "__main__":
    detect_people()
