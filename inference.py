from argparse import ArgumentParser
import utils
import torch
from models.basic_model import CDEvaluator
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.data_utils import CDDataAugmentation

"""
추론 모드를 지원하는 변화 탐지 스크립트

사용법:
1. 학습/평가 모드 (기존 방식, demo.txt 필요):
   python demo_DSIFN_inference.py

2. 추론 모드 (라벨 없이 A, B 폴더만 필요):
   python demo_DSIFN_inference.py --inference --inference_dir ./interference
"""


class InferenceDataset(Dataset):
    """
    추론 전용 데이터셋 클래스
    A, B 폴더의 이미지만 사용하고 라벨은 필요하지 않음
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
            f for f in os.listdir(self.a_dir) 
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ])
        
        if len(self.img_list) == 0:
            raise ValueError(f"A 폴더에 이미지가 없습니다: {self.a_dir}")
        
        # B 폴더에 대응하는 이미지가 있는지 확인
        missing_images = []
        for img_name in self.img_list:
            b_path = os.path.join(self.b_dir, img_name)
            if not os.path.exists(b_path):
                missing_images.append(img_name)
        
        if missing_images:
            raise ValueError(f"B 폴더에 다음 이미지들이 없습니다: {missing_images[:5]}...")
        
        print(f"추론용 이미지 {len(self.img_list)}개를 찾았습니다.")
        
        # 데이터 증강 설정 (추론 모드에서는 증강 없이 리사이즈만)
        self.augm = CDDataAugmentation(img_size=self.img_size)
    
    def __getitem__(self, index):
        """
        데이터셋에서 하나의 샘플을 가져옴
        
        Returns:
            dict: 'name', 'A', 'B' 키를 포함하는 딕셔너리
        """
        img_name = self.img_list[index]
        
        # A, B 이미지 경로
        a_path = os.path.join(self.a_dir, img_name)
        b_path = os.path.join(self.b_dir, img_name)
        
        # 이미지 로드
        img_a = np.asarray(Image.open(a_path).convert('RGB'))
        img_b = np.asarray(Image.open(b_path).convert('RGB'))
        
        # 전처리 (리사이즈 및 텐서 변환)
        [img_a, img_b], _ = self.augm.transform([img_a, img_b], [], to_tensor=True)
        
        return {'name': img_name, 'A': img_a, 'B': img_b}
    
    def __len__(self):
        """데이터셋의 크기 반환"""
        return len(self.img_list)


def get_args():
    """명령줄 인자 파싱"""
    parser = ArgumentParser()
    
    # 기본 설정
    parser.add_argument('--project_name', default='ChangeFormer_slope', type=str,
                       help='프로젝트 이름')
    parser.add_argument('--gpu_ids', type=str, default='0', 
                       help='GPU ID: 예) 0 또는 0,1,2 (CPU는 -1)')
    parser.add_argument('--checkpoint_root', default='./checkpoints_only_for_binary/', type=str,
                       help='체크포인트 루트 디렉토리')
    parser.add_argument('--output_folder', default='samples_DSIFN/predict_ChangeFormerV6', type=str,
                       help='결과 저장 폴더')
    
    # 추론 모드 설정
    parser.add_argument('--inference', action='store_true',
                       help='추론 모드 활성화 (라벨 없이 실행)')
    parser.add_argument('--inference_dir', default='./inference', type=str,
                       help='추론용 데이터 디렉토리 (A, B 폴더 포함)')
    
    # 데이터 설정
    parser.add_argument('--num_workers', default=0, type=int,
                       help='데이터 로더 워커 수')
    parser.add_argument('--dataset', default='CDDataset', type=str,
                       help='데이터셋 타입')
    parser.add_argument('--data_name', default='quick_start_DSIFN', type=str,
                       help='데이터 이름')
    parser.add_argument('--batch_size', default=1, type=int,
                       help='배치 크기')
    parser.add_argument('--split', default="demo", type=str,
                       help='데이터 분할 (train/val/test/demo)')
    parser.add_argument('--img_size', default=256, type=int,
                       help='입력 이미지 크기')
    
    # 모델 설정
    parser.add_argument('--n_class', default=2, type=int,
                       help='클래스 수')
    parser.add_argument('--embed_dim', default=256, type=int,
                       help='임베딩 차원')
    parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
                       help='네트워크 아키텍처')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str,
                       help='체크포인트 파일 이름')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = get_args()
    utils.get_device(args)
    
    # 디바이스 설정
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    
    # 체크포인트 디렉토리 설정
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    
    # 출력 폴더 생성
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 데이터 로더 생성
    if args.inference:
        # 추론 모드: A, B 폴더만 사용
        print(f"[추론 모드] {args.inference_dir}에서 이미지 로드 중...")
        
        # 추론용 데이터셋 생성
        inference_dataset = InferenceDataset(
            inference_dir=args.inference_dir,
            img_size=args.img_size
        )
        
        # 데이터 로더 생성
        data_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        print(f"총 {len(inference_dataset)}개의 이미지 쌍을 처리합니다.")
    else:
        # 기존 모드: demo.txt 파일 사용
        print(f"[평가 모드] {args.data_name} 데이터셋 로드 중...")
        data_loader = utils.get_loader(
            args.data_name, 
            img_size=args.img_size,
            batch_size=args.batch_size,
            split=args.split, 
            is_train=False
        )
    
    # 모델 로드
    print("모델 로드 중...")
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()
    
    # 추론 실행
    print("변화 탐지 시작...")
    for i, batch in enumerate(data_loader):
        name = batch['name']
        print(f'처리 중 ({i+1}/{len(data_loader)}): {name}')
        
        # 순전파 수행
        score_map = model._forward_pass(batch)
        
        # 결과 저장
        model._save_predictions()
    
    print(f"\n완료! 결과가 {args.output_folder}에 저장되었습니다.")