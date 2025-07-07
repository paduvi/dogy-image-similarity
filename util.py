import math
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
import torch
from moviepy import VideoFileClip
from moviepy.config import FFMPEG_BINARY
from torch.utils.data import DataLoader
from tqdm import tqdm

from FrameDataset import FrameDataset


def extract_frames_with_progress(video_path, frame_interval=30, num_threads=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Không mở được video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frames = list(range(0, total_frames, frame_interval))

    # Split work among threads
    chunk_size = len(target_frames) // num_threads
    results = {}

    # Create a shared progress bar
    progress_bar = tqdm(total=len(target_frames), desc="🎬 Trích xuất frame", unit="frame")
    progress_lock = threading.Lock()

    def worker(start_idx, end_idx, thread_id):
        local_cap = cv2.VideoCapture(video_path)
        local_progress = 0

        for i in range(start_idx, end_idx):
            if i >= len(target_frames):
                break
            frame_idx = target_frames[i]
            local_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = local_cap.read()
            if ret:
                timestamp_ms = local_cap.get(cv2.CAP_PROP_POS_MSEC)
                results[i] = (frame, timestamp_ms)

            # Update progress bar safely
            with progress_lock:
                progress_bar.update(1)
                local_progress += 1

        local_cap.release()

    threads = []
    for t in range(num_threads):
        start_idx = t * chunk_size
        end_idx = (t + 1) * chunk_size if t < num_threads - 1 else len(target_frames)
        thread = threading.Thread(target=worker, args=(start_idx, end_idx, t))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    progress_bar.close()

    # Sort results by frame index
    sorted_results = [results[i] for i in sorted(results.keys())]
    frames = [r[0] for r in sorted_results]
    timestamps = [r[1] for r in sorted_results]

    cap.release()
    print("✅ Đã trích xuất xong khung hình.")
    return frames, timestamps, fps


def extract_features_batch(frames, model, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model is on the correct device
    model = model.to(device)

    dataset = FrameDataset(frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    total_batches = len(loader)
    show_progress = len(frames) > 10

    features = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)

            # Debug: Print batch device and model device
            if i == 0:  # Only print for first batch
                print(f"🔧 Batch device: {batch.device}")

            batch_features = model(batch)
            batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
            features.append(batch_features.cpu().numpy())

            if show_progress:
                percent = ((i + 1) / total_batches) * 100
                sys.stdout.write(f"\r🟡 Trích đặc trưng: {percent:.2f}%")
                sys.stdout.flush()

    if show_progress:
        print("\n✅ Trích đặc trưng xong.")

    return np.vstack(features)


def reduce_video_size_and_resolution(input_path, output_path, max_size_gb=2, audio_bitrate_kbps=128,
                                     target_resolution="720p"):
    """
    Giảm kích thước video xuống dưới ngưỡng max_size_gb và đặt lại độ phân giải.
    target_resolution có thể là "720p" (1280x720) hoặc "1080p" (1920x1080).
    """

    # Chuyển đổi kích thước mục tiêu từ GB sang Bytes
    max_size_bytes = max_size_gb * 1000 * 1000 * 1000

    # Định nghĩa độ phân giải mục tiêu
    if target_resolution == "720p":
        target_width = 1280
        target_height = 720
    elif target_resolution == "1080p":
        target_width = 1920
        target_height = 1080
    else:
        print(f"Cảnh báo: Độ phân giải '{target_resolution}' không hợp lệ. Sử dụng 720p làm mặc định.")
        target_width = 1280
        target_height = 720

    # Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file đầu vào '{input_path}'.")
        return None

    current_size = os.path.getsize(input_path)

    # Nếu video đã nhỏ hơn kích thước mục tiêu, bỏ qua việc nén
    if current_size <= max_size_bytes:
        print(
            f"'{input_path}' đã nhỏ hơn {max_size_gb}GB ({current_size / (1024 * 1024):.2f} MB). Bỏ qua việc nén file.")
        return input_path

    print(f"Đang xử lý '{input_path}' (kích thước hiện tại: {current_size / (1024 * 1024):.2f} MB)...")

    clip = VideoFileClip(input_path)
    duration = clip.duration
    clip.close()
    if duration is None or duration == 0:
        print(f"Không thể lấy thời lượng cho '{input_path}'. Bỏ qua.")
        return None

    # Tính toán tổng số bit mục tiêu (giả sử muốn đạt 90% kích thước tối đa để có khoảng trống)
    target_total_bits = (max_size_bytes * 0.90) * 8
    target_total_bitrate_bps = target_total_bits / duration

    audio_bitrate_bps = audio_bitrate_kbps * 1000
    target_video_bitrate_bps = target_total_bitrate_bps - audio_bitrate_bps

    min_video_bitrate_bps = 500 * 1000
    if target_video_bitrate_bps < min_video_bitrate_bps:
        target_video_bitrate_bps = min_video_bitrate_bps

    target_video_bitrate_kbps = math.floor(target_video_bitrate_bps / 1000)

    print(f"Tính toán bitrate video mục tiêu: {target_video_bitrate_kbps} kbps")
    print(f"Độ phân giải mục tiêu: {target_width}x{target_height}")

    # Xây dựng câu lệnh FFmpeg dưới dạng danh sách các chuỗi
    cmd = [
        FFMPEG_BINARY,
        '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', f'{target_video_bitrate_kbps}k',
        '-r', '30',  # Set frame rate to 30 fps
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', f'{audio_bitrate_kbps}k',
        '-vf', f'scale={target_width}:{target_height}',  # Giảm độ phân giải
        '-threads', '0',  # Sử dụng tất cả các luồng CPU có sẵn
        '-y',  # Ghi đè nếu file đầu ra đã tồn tại
        output_path
    ]

    if run_ffmpeg_cmd(cmd):
        print(f"\nHoàn thành xử lý '{input_path}'. File đầu ra: '{output_path}'")
        final_size = os.path.getsize(output_path)
        print(f"Kích thước file đầu ra: {final_size / (1024 * 1024):.2f} MB")
        return output_path
    return None


def run_ffmpeg_cmd(cmd):
    try:
        # In câu lệnh ra console để kiểm tra
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi: FFmpeg đã thoát với mã lỗi {e.returncode}. Kiểm tra thông báo trên console để biết chi tiết.")
        return False
    except FileNotFoundError:
        print("❌ Lỗi: FFmpeg không tìm thấy. Hãy đảm bảo nó đã được cài đặt và có trong biến môi trường PATH của bạn.")
        return False


def cut_video(input_path, output_path, start_time, duration=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    cmd = [
        FFMPEG_BINARY,
        '-i', input_path,
        '-c', 'copy',
        '-threads', '0',  # Sử dụng tất cả các luồng CPU có sẵn
        '-y',  # Ghi đè nếu file đầu ra đã tồn tại
    ]

    if duration is not None:
        cmd.extend([
            '-ss', str(math.floor(start_time)),
            '-t', str(duration),
            output_path
        ])
    else:
        cmd.extend([
            '-ss', str(math.floor(start_time)),
            output_path
        ])
    if run_ffmpeg_cmd(cmd):
        print(f"\nHoàn thành xử lý '{input_path}'. File đầu ra: '{output_path}'")
