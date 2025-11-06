from argparse import ArgumentParser
import utils
import torch
import torch.nn as nn
from models.basic_model import CDEvaluator
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.data_utils import CDDataAugmentation
import gc

"""
Jetson Orin Nano용 메모리 최적화 추론 스크립트

최적화 사항:
1. Mixed Precision (FP16) 사용
2. 배치 크기 1 고정
3. 메모리 효율적인 설정
4. 불필요한 중간 출력 제거
5. 작은 embedding dimension 옵션

사용법:
python inference_jetson_optimized.py \
    --checkpoint_path ./checkpoints/best_ckpt.pt \
    --inference_dir ./inference \
    --output_folder ./output \
    --use_fp16 \
    --embed_dim 128
"""

class InferenceDataset(Dataset):
    """
    추론 전용 데이터셋 클래스
    메모리 효율적으로 이미지를 로드
    """
    def __init__(self, inference_dir, img_size=256):
        """
        Args:
            inference_dir: A, B 폴더가 있는 상위 디렉토리 경로
            img_size: 이미지 크기
        """
        self.inference_dir = inference_dir
        self.img_size = img_size
        
        # A, B 폴더 경로 설정
        self.a_dir = os.path.join(inference_dir, 'A')
        self.b_dir = os.path.join(inference_dir, 'B')
        
        # 폴더 존재 확인
        if not os.path.exists(self.a_dir):
            raise ValueError(f"A 폴더를 찾을 수 없습니다: {self.a_dir}")
        if not os.path.exists(self.b_dir):
            raise ValueError(f"B 폴더를 찾을 수 없습니다: {self.b_dir}")
        
        # A 폴더의 이미지 파일 리스트 가져오기
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        self.img_list = sorted([
            str(f) for f in os.listdir(self.a_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ])
        
        if len(self.img_list) == 0:
            raise ValueError(f"A 폴더에 이미지가 없습니다: {self.a_dir}")
        
        print(f"추론용 이미지 {len(self.img_list)}개를 찾았습니다.")
        
        # 데이터 변환 설정
        self.transform = CDDataAugmentation(img_size=img_size)
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """
        단일 이미지 쌍 로드 및 전처리
        메모리 효율적으로 처리
        """
        img_name = self.img_list[idx]
        
        # 이미지 경로
        a_path = os.path.join(self.a_dir, img_name)
        b_path = os.path.join(self.b_dir, img_name)
        
        # 이미지 로드 (RGB로 변환)
        img_a = np.array(Image.open(a_path).convert('RGB'))
        img_b = np.array(Image.open(b_path).convert('RGB'))
        
        # 데이터 변환 (텐서로 변환 및 정규화)
        [img_a, img_b], _ = self.transform.transform([img_a, img_b], [], to_tensor=True)
        
        # 파일명에서 확장자 제거
        name = os.path.splitext(img_name)[0]
        
        return {
            'A': img_a,
            'B': img_b,
            'name': name
        }


class OptimizedCDEvaluator(CDEvaluator):
    """
    메모리 최적화된 Change Detection Evaluator
    """
    
    def __init__(self, args):
        """
        초기화 시 메모리 최적화 설정 적용
        """
        super().__init__(args)
        self.use_fp16 = args.use_fp16
        
        # FP16 사용 시 모델 변환
        if self.use_fp16:
            print("FP16 (Mixed Precision) 모드 활성화")
            self.net_G = self.net_G.half()
    
    def _forward_pass(self, batch):
        """
        메모리 효율적인 순전파
        
        Args:
            batch: 입력 배치 딕셔너리
        
        Returns:
            pred: 예측 마스크 (numpy array)
        """
        # 입력 데이터를 GPU로 이동
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        # FP16 변환 (필요한 경우)
        if self.use_fp16:
            img_in1 = img_in1.half()
            img_in2 = img_in2.half()
        
        # Gradient 계산 비활성화 (추론 모드)
        with torch.no_grad():
            # 자동 Mixed Precision 사용
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                # 모델 순전파
                outputs = self.net_G(img_in1, img_in2)
                
                # 마지막 출력만 사용 (메모리 절약)
                # outputs는 다중 스케일 출력 리스트
                pred = outputs[-1]
        
        # 예측 결과를 이진 마스크로 변환
        pred = pred.argmax(dim=1)  # [B, H, W]
        
        # CPU로 이동하고 numpy 배열로 변환
        pred_np = pred.cpu().numpy()
        
        # GPU 메모리 정리
        del img_in1, img_in2, outputs, pred
        torch.cuda.empty_cache()
        
        return pred_np
    
    def _save_predictions(self, pred_np, name, output_folder):
        """
        예측 결과를 이미지로 저장
        
        Args:
            pred_np: 예측 마스크 numpy 배열
            name: 파일명
            output_folder: 출력 폴더 경로
        """
        # 배치 차원 제거
        if pred_np.ndim == 3:
            pred_np = pred_np[0]
        
        # 0-255 범위로 스케일링
        pred_vis = (pred_np * 255).astype(np.uint8)
        
        # PIL 이미지로 변환 및 저장
        pred_img = Image.fromarray(pred_vis)
        save_path = os.path.join(output_folder, f'{name}.png')
        pred_img.save(save_path)


def get_args():
    """
    명령줄 인자 파싱
    """
    parser = ArgumentParser(description='Jetson Orin Nano 최적화 추론')
    
    # 필수 인자
    parser.add_argument('--checkpoint_path', default='./checkpoints_only_for_binary/ChangeFormer_slope/best_ckpt.pt', type=str,
                       help='체크포인트 파일 전체 경로 (예: ./checkpoints/project/best_ckpt.pt)')
    parser.add_argument('--inference_dir',default='./inference', type=str,
                       help='추론할 이미지가 있는 디렉토리 (A, B 폴더 포함)')
    parser.add_argument('--output_folder', default='./inference/output', type=str,
                       help='결과 저장 폴더')
    
    # GPU 설정
    parser.add_argument('--gpu_ids', type=str, default='0',
                       help='사용할 GPU ID (예: 0 또는 0,1)')
    
    # 데이터 로더 설정
    parser.add_argument('--num_workers', default=2, type=int,
                       help='데이터 로더 워커 수 (Jetson에서는 2-4 권장)')
    parser.add_argument('--batch_size', default=1, type=int,
                       help='배치 크기')
    parser.add_argument('--img_size', default=256, type=int,
                       help='입력 이미지 크기')
    
    # 모델 설정
    parser.add_argument('--n_class', default=2, type=int,
                       help='클래스 수')
    parser.add_argument('--embed_dim', default=256, type=int,
                       help='임베딩 차원')
    parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
                       help='네트워크 아키텍처')
    
    # 최적화 옵션
    parser.add_argument('--use_fp16', action='store_true',
                       help='FP16 (Mixed Precision) 사용 (메모리 50% 절약)')
    parser.add_argument('--pin_memory', action='store_true',
                       help='Pin memory 사용 (전송 속도 향상, 메모리 약간 증가)')
    
    args = parser.parse_args()
    
    # 체크포인트 경로에서 디렉토리와 파일명 분리
    args.checkpoint_dir = os.path.dirname(args.checkpoint_path)
    args.checkpoint_name = os.path.basename(args.checkpoint_path)
    args.project_name = os.path.basename(args.checkpoint_dir)
    
    return args


def print_memory_usage():
    """
    현재 GPU 메모리 사용량 출력
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB")


def main():
    """
    메인 추론 함수
    """
    # 인자 파싱
    args = get_args()
    
    # GPU 설정
    utils.get_device(args)
    device = torch.device(f"cuda:{args.gpu_ids[0]}" 
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    
    print(f"사용 디바이스: {device}")
    print(f"FP16 사용: {args.use_fp16}")
    print(f"Embedding 차원: {args.embed_dim}")
    print(f"이미지 크기: {args.img_size}")
    
    # 출력 폴더 생성
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 데이터셋 생성
    print(f"\n[추론 모드] {args.inference_dir}에서 이미지 로드 중...")
    inference_dataset = InferenceDataset(
        inference_dir=args.inference_dir,
        img_size=args.img_size
    )
    
    # 데이터 로더 생성 (메모리 효율적 설정)
    data_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    
    print(f"총 {len(inference_dataset)}개의 이미지 쌍을 처리합니다.")
    
    # 모델 로드
    print("\n모델 로드 중...")
    model = OptimizedCDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()
    
    print_memory_usage()
    
    # 추론 실행
    print("\n변화 탐지 시작...")
    for i, batch in enumerate(data_loader):
        name = batch['name']
        
        # 이름을 문자열로 변환
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        
        print(f'처리 중 ({i+1}/{len(data_loader)}): {name_str}')
        
        # 순전파 수행
        pred_np = model._forward_pass(batch)
        
        # 결과 저장
        model._save_predictions(pred_np, name_str, args.output_folder)
        
        # 주기적으로 메모리 사용량 출력
        if (i + 1) % 10 == 0:
            print_memory_usage()
            # 가비지 컬렉션 실행
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n완료! 결과가 {args.output_folder}에 저장되었습니다.")
    print_memory_usage()


if __name__ == '__main__':
    main()