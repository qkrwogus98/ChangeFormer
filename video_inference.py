import argparse
import os
from collections import deque
import cv2
import numpy as np
import torch
from PIL import Image

# Modules from ChangeFormer project
import utils
from models.basic_model import CDEvaluator
from datasets.data_utils import CDDataAugmentation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run change detection on a video")
    parser.add_argument('--input_video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output masks')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Model checkpoint (.pt)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma separated GPU ids')
    parser.add_argument('--interval_sec', type=float, default=5.0,
                        help='Time interval between frame pairs in seconds')

    parser.add_argument('--net_G', default='ChangeFormerV6', type=str)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--n_class', default=2, type=int)

    args = parser.parse_args()
    args.checkpoint_dir, args.checkpoint_name = os.path.split(args.checkpoint_path)
    args.project_name = os.path.basename(args.checkpoint_dir)
    args.output_folder = args.output_dir
    return args


def preprocess_frame(frame, transform):
    """Convert an OpenCV frame to a model-ready tensor."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    [img_tensor], _ = transform.transform([np.array(img_pil)], [], to_tensor=True)
    return img_tensor.unsqueeze(0)


def main():
    args = parse_args()

    gpu_ids = [int(i) for i in args.gpu_ids.split(',') if i]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() and gpu_ids else 'cpu')
    utils.get_device(args)

    os.makedirs(args.output_dir, exist_ok=True)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)

    # move the underlying network to the desired device if possible
    if hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass
    if not hasattr(model, "to") or isinstance(model, torch.nn.Module) is False:
        if hasattr(model, "model"):
            model.model.to(device)
        elif hasattr(model, "net_G"):
            model.net_G.to(device)

    model.eval()

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.input_video}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_offset = max(1, int(args.interval_sec * fps))
    print(
        f"Video: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames; "
        f"processing every {args.interval_sec} seconds ({frame_offset} frames)"
    )

    transform = CDDataAugmentation(img_size=args.img_size)

    buffer = deque(maxlen=frame_offset + 1)
    frame_index = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer.append(frame)

        if len(buffer) < frame_offset + 1:
            frame_index += 1
            continue

        frame_t1 = buffer[0]
        frame_t2 = buffer[-1]

        tensor_t1 = preprocess_frame(frame_t1, transform).to(device)
        tensor_t2 = preprocess_frame(frame_t2, transform).to(device)
        batch = {'A': tensor_t1, 'B': tensor_t2, 'name': [f'frame_{saved}']}

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            try:
                pred_mask = model(batch)
            except Exception:
                pred_mask = model._forward_pass(batch)

        mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        timestamp_sec = frame_index / fps
        minutes = int(timestamp_sec // 60)
        seconds = int(timestamp_sec % 60)
        filename = f'{minutes:02d}분{seconds:02d}초.png'
        cv2.imwrite(os.path.join(args.output_dir, filename), mask)

        saved += 1
        if saved % 10 == 0:
            print(f"Processed {saved} frames")

        frame_index += 1

    cap.release()
    print(f"Finished! Saved results to {args.output_dir}")


if __name__ == '__main__':
    main()