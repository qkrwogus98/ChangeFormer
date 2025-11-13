"""
실시간 RTSP 카메라 스트림에서 변화 탐지를 수행하는 스크립트

사용법:
1. camera_config.yaml 파일을 생성하여 카메라 정보 입력
2. python camera_inference.py --config camera_config.yaml
"""

import argparse
import os
import time
from collections import deque
from datetime import datetime
import yaml
import cv2
import numpy as np
import torch

# ChangeFormer 프로젝트 모듈
import utils
from models.basic_model import CDEvaluator
from datasets.data_utils import CDDataAugmentation


def parse_args():
    """
    명령줄 인자를 파싱하는 함수
    
    Returns:
        Namespace: 파싱된 인자들
    """
    parser = argparse.ArgumentParser(description="실시간 RTSP 카메라 스트림에서 변화 탐지")
    
    # 설정 파일
    parser.add_argument('--config', type=str, required=True, 
                       help='카메라 설정 YAML 파일 경로 (예: camera_config.yaml)')
    
    # 모델 설정
    parser.add_argument('--checkpoint_path', type=str, 
                       default='./pretrained/ChangeFormer_slope/best_ckpt.pt',
                       help='모델 체크포인트 파일 경로')
    parser.add_argument('--gpu_ids', type=str, default='0', 
                       help='사용할 GPU ID (쉼표로 구분, CPU는 -1)')
    
    # 추론 설정
    parser.add_argument('--interval_sec', type=float, default=1.0,
                       help='프레임 비교 간격 (초 단위)')
    parser.add_argument('--img_size', default=256, type=int,
                       help='모델 입력 이미지 크기')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
                       help='네트워크 아키텍처')
    parser.add_argument('--embed_dim', default=256, type=int,
                       help='임베딩 차원')
    parser.add_argument('--n_class', default=2, type=int,
                       help='클래스 수')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='./camera_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='저장 진행 상황을 출력할 프레임 간격')
    
    args = parser.parse_args()
    
    # 체크포인트 경로 파싱
    args.checkpoint_dir, args.checkpoint_name = os.path.split(args.checkpoint_path)
    args.project_name = os.path.basename(args.checkpoint_dir)
    args.output_folder = args.output_dir
    
    return args


def load_camera_config(config_path):
    """
    YAML 설정 파일에서 카메라 정보를 로드하는 함수
    
    Args:
        config_path (str): YAML 설정 파일 경로
        
    Returns:
        dict: 카메라 설정 정보
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def connect_rtsp_stream(rtsp_url, retry_attempts=3, retry_delay=2):
    """
    RTSP 스트림에 연결하는 함수 (재시도 로직 포함, 최적화됨)

    Args:
        rtsp_url (str): RTSP 스트림 URL
        retry_attempts (int): 재시도 횟수
        retry_delay (int): 재시도 간 대기 시간 (초)

    Returns:
        cv2.VideoCapture: 비디오 캡처 객체

    Raises:
        RuntimeError: 연결 실패 시
    """
    for attempt in range(retry_attempts):
        print(f"카메라 연결 시도 중... ({attempt + 1}/{retry_attempts})")
        cap = cv2.VideoCapture(rtsp_url)

        if cap.isOpened():
            # 버퍼 크기 최소화로 지연 시간 감소
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("✓ 카메라 연결 성공!")
            return cap

        print(f"✗ 연결 실패. {retry_delay}초 후 재시도...")
        time.sleep(retry_delay)

    raise RuntimeError(f"카메라 연결 실패: {rtsp_url}")


def preprocess_frame(frame, transform, device):
    """
    OpenCV 프레임을 모델 입력 텐서로 변환하는 함수 (최적화됨)

    Args:
        frame (numpy.ndarray): OpenCV BGR 프레임
        transform (CDDataAugmentation): 데이터 변환 객체
        device (torch.device): 타겟 디바이스

    Returns:
        torch.Tensor: 전처리된 텐서 (1, C, H, W)
    """
    # BGR을 RGB로 변환 (불필요한 PIL 변환 제거)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 데이터 증강 및 텐서 변환
    [img_tensor], _ = transform.transform([img_rgb], [], to_tensor=True)

    # 배치 차원 추가 후 바로 device로 이동
    return img_tensor.unsqueeze(0).to(device, non_blocking=True)


def save_change_mask(mask, output_dir, frame_count, original_size):
    """
    변화 탐지 결과를 이미지 파일로 저장하는 함수 (최적화됨)

    Args:
        mask (numpy.ndarray): 변화 탐지 마스크
        output_dir (str): 출력 디렉토리
        frame_count (int): 프레임 카운터
        original_size (tuple): 원본 프레임 크기 (width, height)

    Returns:
        tuple: (파일명, 타임스탬프)
    """
    # 타임스탬프 생성
    timestamp = datetime.now()

    # 원본 크기로 리사이즈
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # 파일명 생성 (날짜_시간_프레임번호.png)
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{time_str}_frame{frame_count:06d}.png"
    filepath = os.path.join(output_dir, filename)

    # 저장
    cv2.imwrite(filepath, mask_resized)

    return filename, timestamp


def main():
    """메인 함수"""
    
    # 인자 파싱
    args = parse_args()
    
    # 카메라 설정 로드
    print(f"설정 파일 로드 중: {args.config}")
    camera_config = load_camera_config(args.config)
    rtsp_url = camera_config['rtsp_url']
    
    print(f"RTSP URL: {rtsp_url}")
    
    # GPU 설정
    gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i.strip() and i != '-1']
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() and gpu_ids else 'cpu')
    print(f"사용 디바이스: {device}")
    
    utils.get_device(args)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"결과 저장 경로: {args.output_dir}")
    
    # 모델 로드
    print("\n모델 로드 중...")
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)

    # 모델을 디바이스로 이동 및 평가 모드 설정
    if hasattr(model, 'net_G'):
        model.net_G.to(device)
        model.net_G.eval()
        # 그래디언트 계산 완전히 비활성화 (추론 성능 향상)
        for param in model.net_G.parameters():
            param.requires_grad = False

    print("✓ 모델 로드 완료")
    
    # RTSP 스트림 연결
    print(f"\nRTSP 스트림 연결 중...")
    cap = connect_rtsp_stream(rtsp_url)
    
    # 스트림 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # FPS가 0이거나 비정상적인 경우 기본값 설정
    if fps <= 0 or fps > 120:
        fps = 25.0  # 기본값
        print(f"⚠ FPS 정보를 가져올 수 없어 기본값({fps})을 사용합니다.")
    
    print(f"스트림 정보: {frame_width}x{frame_height}, {fps:.2f} FPS")
    
    # 프레임 간격 계산 (1초 간격이면 fps만큼의 프레임)
    frame_offset = max(1, int(args.interval_sec * fps))
    print(f"처리 간격: {args.interval_sec}초마다 ({frame_offset} 프레임)")
    
    # 데이터 전처리 준비
    transform = CDDataAugmentation(img_size=args.img_size)

    # 프레임 버퍼 (deque를 사용하여 메모리 효율적으로 관리)
    buffer = deque(maxlen=frame_offset + 1)

    # 카운터 및 캐시 초기화
    saved_count = 0
    cached_tensor_t1 = None  # 이전 프레임 텐서 캐싱

    print(f"\n{'='*60}")
    print("변화 탐지 시작!")
    print(f"{'='*60}\n")

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()

            if not ret:
                print("⚠ 프레임을 읽을 수 없습니다. 스트림이 종료되었거나 연결이 끊어졌습니다.")
                break

            # 버퍼에 프레임 추가
            buffer.append(frame)

            # 버퍼가 충분히 차지 않았으면 계속 읽기
            if len(buffer) < frame_offset + 1:
                continue

            # 비교할 두 프레임 가져오기 (첫 번째와 마지막)
            frame_t2 = buffer[-1]

            # 텐서 전처리 (캐싱 활용)
            if cached_tensor_t1 is None:
                # 첫 번째 실행 시에만 t1 전처리
                tensor_t1 = preprocess_frame(buffer[0], transform, device)
            else:
                # 이전에 전처리한 t2를 t1으로 재사용
                tensor_t1 = cached_tensor_t1

            tensor_t2 = preprocess_frame(frame_t2, transform, device)

            # 다음 반복을 위해 t2 캐싱
            cached_tensor_t1 = tensor_t2

            # 배치 준비
            batch = {'A': tensor_t1, 'B': tensor_t2}

            # 추론 수행 (inference_mode는 no_grad보다 빠름)
            with torch.inference_mode():
                # GPU 사용 시 자동 혼합 정밀도 활성화
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    try:
                        # 모델 순전파
                        pred_mask = model._forward_pass(batch)
                    except Exception as e:
                        print(f"⚠ 추론 중 오류 발생: {e}")
                        continue

            # 마스크 후처리 (CPU로 이동 후 numpy 변환)
            mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

            # GPU 메모리 정리
            del pred_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # 결과 저장
            filename, timestamp = save_change_mask(
                mask,
                args.output_dir,
                saved_count,
                (frame_width, frame_height)
            )

            saved_count += 1

            # 진행 상황 출력
            if saved_count % args.save_interval == 0:
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"처리 완료: {saved_count}개 프레임 | 최근 파일: {filename}")
    
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    
    except Exception as e:
        print(f"\n⚠ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 정리
        cap.release()
        print(f"\n{'='*60}")
        print(f"처리 완료!")
        print(f"총 {saved_count}개의 결과가 저장되었습니다.")
        print(f"저장 경로: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()