import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import torch
from torchvision.models import resnet18

from util import extract_features_batch, extract_frames_with_progress, cut_video


def select_files():
    """
    Opens a file dialog, allows multiple file selection,
    and returns a list of selected file paths.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(
        title="Select Input Files",
        initialdir=os.getcwd() + os.sep + "input",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
        )
    )

    # file_paths will be a tuple of strings
    return list(file_paths)


def find_sample_end_moment(similarity_matrix, timestamps, threshold=0.9, min_high_duration=2.0):
    """
    Find the moment when similarity score drops after staying high.

    Args:
        similarity_matrix: (N_frames, 1) similarity scores
        timestamps: frame timestamps in milliseconds
        threshold: minimum similarity score to consider "high"
        min_high_duration: minimum duration (seconds) the score should stay high

    Returns:
        (frame_idx, timestamp_seconds) of the drop moment, or None if not found
    """
    similarities = similarity_matrix[:, 0]  # Get first (and only) column

    # Find segments where similarity is above threshold
    high_mask = similarities >= threshold

    # Find transitions (start/end of high periods)
    transitions = np.diff(high_mask.astype(int))
    high_starts = np.where(transitions == 1)[0] + 1  # Start of high period
    high_ends = np.where(transitions == -1)[0] + 1  # End of high period

    # Handle edge cases
    if len(high_starts) == 0 and len(high_ends) == 0:
        return None  # No high periods found

    # If starts with high period
    if len(high_ends) > 0 and (len(high_starts) == 0 or high_ends[0] < high_starts[0]):
        high_starts = np.concatenate([[0], high_starts])

    # If ends with high period
    if len(high_starts) > 0 and (len(high_ends) == 0 or high_starts[-1] > high_ends[-1]):
        high_ends = np.concatenate([high_ends, [len(similarities)]])

    # Find high periods that last long enough
    for start_idx, end_idx in zip(high_starts, high_ends):
        if start_idx >= len(timestamps) or end_idx >= len(timestamps):
            continue

        duration = (timestamps[end_idx] - timestamps[start_idx]) / 1000.0  # Convert to seconds

        if duration >= min_high_duration:
            # Found a valid high period, return the drop moment
            drop_frame_idx = end_idx
            drop_timestamp = timestamps[drop_frame_idx] / 1000.0
            return drop_frame_idx, drop_timestamp

    return None


# Replace the existing steps 5-6 with this new implementation:
def process_file(input_path):
    file_name = os.path.basename(input_path)

    # 1. Trích frame & timestamps
    start = time.time()
    frames, timestamps, fps = extract_frames_with_progress(input_path, frame_interval=30)
    end = time.time()
    print(f"\n⚡ Số lượng frame thu được: {len(frames)}")
    print(f"\n🕒 [1] Trích xuất frame: {end - start:.2f} giây")

    # 2. Trích đặc trưng theo batch
    start = time.time()
    features = extract_features_batch(frames, model=model)  # shape: (N_frames, D)
    end = time.time()
    print(f"🕒 [2] Trích đặc trưng frame: {end - start:.2f} giây")

    # 3. Trích đặc trưng ảnh mẫu
    start = time.time()
    query_image = "sample.png"
    query_features = extract_features_batch([cv2.imread(query_image)], model=model)  # shape: (1, D)
    end = time.time()
    print(f"🕒 [3] Trích đặc trưng ảnh mẫu: {end - start:.2f} giây")

    # 4. Tính độ tương đồng cosine (dot product vì đã normalize)
    start = time.time()
    similarity_matrix = np.dot(features, query_features.T)  # shape: (N_frames, 1)
    end = time.time()
    print(f"🕒 [4] Tính độ tương đồng cosine: {end - start:.2f} giây")

    # 5. Tìm thời điểm kết thúc mẫu (khi similarity score giảm sau khi cao)
    start = time.time()
    result = find_sample_end_moment(similarity_matrix, timestamps, threshold=0.9, min_high_duration=2.0)
    end = time.time()
    print(f"🕒 [5] Tìm thời điểm kết thúc mẫu: {end - start:.2f} giây")

    if result is None:
        print("❌ Không tìm thấy thời điểm kết thúc mẫu phù hợp")
        return

    chosen_idx, chosen_time = result
    chosen_sim = similarity_matrix[chosen_idx, 0]

    print(f"✅ Frame được chọn: {chosen_idx:03d} @ {chosen_time:.2f}s (similarity: {chosen_sim:.4f})")

    # 6. Cắt video tính từ thời điểm được chọn
    start = time.time()
    output_path = os.path.join("output", "resnet18", file_name)
    cut_video(input_path, output_path, chosen_time)
    end = time.time()
    print(f"🕒 [6] Cắt video: {end - start:.2f} giây")


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # --- Chuẩn bị mô hình ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights='DEFAULT')
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()

    print("Starting file selection...")
    selected_files = select_files()

    if selected_files:
        print(f"\n{len(selected_files)} file(s) selected:")
        for i, file_path in enumerate(selected_files):
            print(f"\n+ {file_path}")
            process_file(file_path)
    else:
        print("No files were selected.")
