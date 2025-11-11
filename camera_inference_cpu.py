"""
실시간 RTSP 카메라 스트림에서 변화 탐지를 수행하는 스크립트
CPU 및 GPU 환경 모두 지원

사용법:
1. camera_config.yaml 파일을 생성하여 카메라 정보 입력
2. python camera_inference.py --config camera_config.yaml

python camera_inference_cpu.py --config camera_config.yaml --gpu_ids -1

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
from PIL import Image

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
    RTSP 스트림에 연결하는 함수 (재시도 로직 포함)
    
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
            print("✓ 카메라 연결 성공!")
            return cap
        
        print(f"✗ 연결 실패. {retry_delay}초 후 재시도...")
        time.sleep(retry_delay)
    
    raise RuntimeError(f"카메라 연결 실패: {rtsp_url}")


def preprocess_frame(frame, transform):
    """
    OpenCV 프레임을 모델 입력 텐서로 변환하는 함수
    
    Args:
        frame (numpy.ndarray): OpenCV BGR 프레임
        transform (CDDataAugmentation): 데이터 변환 객체
        
    Returns:
        torch.Tensor: 전처리된 텐서 (1, C, H, W)
    """
    # BGR을 RGB로 변환
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # PIL 이미지로 변환 후 numpy 배열로
    img_pil = Image.fromarray(img_rgb)
    img_np = np.array(img_pil)
    
    # 데이터 증강 및 텐서 변환
    [img_tensor], _ = transform.transform([img_np], [], to_tensor=True)
    
    # 배치 차원 추가
    return img_tensor.unsqueeze(0)


def save_change_mask(mask, output_dir, timestamp, frame_count, original_size):
    """
    변화 탐지 결과를 이미지 파일로 저장하는 함수
    
    Args:
        mask (numpy.ndarray): 변화 탐지 마스크
        output_dir (str): 출력 디렉토리
        timestamp (datetime): 현재 타임스탬프
        frame_count (int): 프레임 카운터
        original_size (tuple): 원본 프레임 크기 (width, height)
        
    Returns:
        str: 저장된 파일명
    """
    # 원본 크기로 리사이즈
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # 파일명 생성 (날짜_시간_프레임번호.png)
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{time_str}_frame{frame_count:06d}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 저장
    cv2.imwrite(filepath, mask_resized)
    
    return filename


def setup_device(args):
    """
    사용할 디바이스(CPU/GPU)를 설정하는 함수
    CPU 환경에서도 안정적으로 작동하도록 개선
    
    Args:
        args: 명령줄 인자
        
    Returns:
        torch.device: 사용할 디바이스
        bool: GPU 사용 여부
    """
    # GPU ID 파싱
    gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i.strip() and i != '-1']
    
    # CUDA 사용 가능 여부 확인
    use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
    
    if use_gpu:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"✓ GPU 사용: {device}")
        print(f"  - GPU 이름: {torch.cuda.get_device_name(gpu_ids[0])}")
        print(f"  - 메모리: {torch.cuda.get_device_properties(gpu_ids[0]).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        if len(gpu_ids) > 0:
            print(f"⚠ CUDA를 사용할 수 없어 CPU를 사용합니다.")
        else:
            print(f"✓ CPU 사용")
        print(f"  - CPU 스레드 수: {torch.get_num_threads()}")
    
    # utils.get_device 호출 (기존 코드 호환성)
    utils.get_device(args)
    
    return device, use_gpu


def load_model_safe(args, device):
    """
    모델을 안전하게 로드하는 함수 (CPU/GPU 모두 지원)
    
    Args:
        args: 명령줄 인자
        device: 사용할 디바이스
        
    Returns:
        CDEvaluator: 로드된 모델
    """
    print("\n모델 로드 중...")
    
    # 모델 생성
    model = CDEvaluator(args)
    
    # 체크포인트 로드 (CPU 환경 고려)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    # CPU로 로드할지 결정
    if device.type == 'cpu':
        # CPU에서 로드 시 map_location 지정
        print("  - CPU 환경에서 체크포인트 로드 중...")
        model.load_checkpoint(args.checkpoint_name)
    else:
        # GPU에서 로드
        print(f"  - GPU 환경에서 체크포인트 로드 중...")
        model.load_checkpoint(args.checkpoint_name)
    
    # 모델을 디바이스로 이동
    if hasattr(model, 'net_G'):
        model.net_G.to(device)
        print(f"  - 모델을 {device}로 이동 완료")
    
    # 평가 모드로 설정
    model.eval()
    print("✓ 모델 로드 완료")
    
    return model


def main():
    """메인 함수"""
    
    # 인자 파싱
    args = parse_args()
    
    # 카메라 설정 로드
    print(f"설정 파일 로드 중: {args.config}")
    camera_config = load_camera_config(args.config)
    rtsp_url = camera_config['rtsp_url']
    
    print(f"RTSP URL: {rtsp_url}")
    
    # 디바이스 설정 (CPU/GPU 자동 선택)
    device, use_gpu = setup_device(args)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"결과 저장 경로: {args.output_dir}")
    
    # 모델 로드 (CPU/GPU 안전 로드)
    model = load_model_safe(args, device)
    
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
    
    # 카운터 초기화
    frame_index = 0
    saved_count = 0
    
    print(f"\n{'='*60}")
    print("변화 탐지 시작!")
    print(f"디바이스: {device}")
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
            frame_index += 1
            
            # 버퍼가 충분히 차지 않았으면 계속 읽기
            if len(buffer) < frame_offset + 1:
                continue
            
            # 비교할 두 프레임 가져오기 (첫 번째와 마지막)
            frame_t1 = buffer[0]
            frame_t2 = buffer[-1]
            
            # 프레임 전처리
            tensor_t1 = preprocess_frame(frame_t1, transform).to(device)
            tensor_t2 = preprocess_frame(frame_t2, transform).to(device)
            
            # 배치 준비
            batch = {
                'A': tensor_t1, 
                'B': tensor_t2, 
                'name': [f'frame_{saved_count}']
            }
            
            # 추론 수행 (CPU/GPU 모두 안정적으로 작동)
            with torch.no_grad():
                try:
                    # GPU 사용 시에만 자동 혼합 정밀도 활성화
                    # CPU에서는 AMP를 사용하지 않음
                    if use_gpu:
                        with torch.cuda.amp.autocast():
                            pred_mask = model._forward_pass(batch)
                    else:
                        # CPU에서는 일반 추론
                        pred_mask = model._forward_pass(batch)
                        
                except Exception as e:
                    print(f"⚠ 추론 중 오류 발생: {e}")
                    continue
            
            # 마스크 후처리 (결과가 스칼라인 경우 처리)
            if isinstance(pred_mask, torch.Tensor):
                mask = pred_mask.squeeze().cpu().numpy()
            else:
                mask = pred_mask
            
            # numpy 배열로 변환 및 타입 확인
            mask = np.array(mask, dtype=np.uint8)
            
            # 결과 저장
            timestamp = datetime.now()
            filename = save_change_mask(
                mask, 
                args.output_dir, 
                timestamp, 
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
