import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18

from util import extract_features_batch, extract_frames_with_progress


def create_non_blocking_plot(video_name, timestamps, similarity_scores):
    """
    Creates a non-blocking plot for similarity scores over time.
    """
    # Create a new figure for each video
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up the plot
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Similarity Score')
    ax.set_title(f'Similarity Score Over Time - {video_name}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)  # Assuming similarity scores are between 0 and 1

    # Convert timestamps to seconds
    time_seconds = [ts / 1000.0 for ts in timestamps]  # Assuming timestamps are in milliseconds

    # Plot the similarity scores
    line, = ax.plot(time_seconds, similarity_scores, 'b-', linewidth=2, label='Similarity Score')

    # Add some statistics
    avg_score = np.mean(similarity_scores)
    max_score = np.max(similarity_scores)
    max_time = time_seconds[np.argmax(similarity_scores)]

    ax.axhline(y=avg_score, color='r', linestyle='--', alpha=0.7, label=f'Average: {avg_score:.3f}')
    ax.axvline(x=max_time, color='g', linestyle='--', alpha=0.7, label=f'Max at {max_time:.1f}s: {max_score:.3f}')

    ax.legend()

    # Make the plot non-blocking
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)

    # Keep the plot window responsive
    plt.pause(0.1)

    return fig



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
    features = extract_features_batch(frames, model=model)  # shape: (M, D)
    end = time.time()
    print(f"🕒 [2] Trích đặc trưng frame: {end - start:.2f} giây")

    # 3. Trích đặc trưng ảnh mẫu
    start = time.time()
    query_image = "sample.png"
    query_features = extract_features_batch([cv2.imread(query_image)], model=model)  # shape: (1, D)
    end = time.time()
    print(f"🕒 [3] Trích đặc trưng ảnh mẫu: {end - start:.2f} giây")

    # 4. Tính độ tương đồng cosine (dot product vì đã normalize)
    #    features: (N_frames, D), query_features.T:
    start = time.time()
    similarity_matrix = np.dot(features, query_features.T)  # shape: (M, 1)
    similarity_scores = similarity_matrix.flatten()  # Convert to 1D array
    end = time.time()
    print(f"🕒 [4] Tính độ tương đồng cosine: {end - start:.2f} giây")

    # 5. Create non-blocking plot
    print(f"📊 Creating plot for {file_name}...")
    fig = create_non_blocking_plot(file_name, timestamps, similarity_scores)

    # Print some statistics
    print(f"📈 Statistics for {file_name}:")
    print(f"   Average similarity: {np.mean(similarity_scores):.3f}")
    print(f"   Max similarity: {np.max(similarity_scores):.3f}")
    print(f"   Min similarity: {np.min(similarity_scores):.3f}")
    print(f"   Std deviation: {np.std(similarity_scores):.3f}")


def process_files_with_plots(selected_files):
    """
    Process files and create plots in a way that allows interaction with plots
    while processing continues.
    """
    for i, file_path in enumerate(selected_files):
        print(f"\n{'=' * 60}")
        print(f"Processing file {i + 1}/{len(selected_files)}")
        print(f"{'=' * 60}")

        # Process the file and create plot
        process_file(file_path)

        # Small delay to allow plot to render
        time.sleep(0.5)

        # Process events to keep plots responsive
        if plt.get_fignums():  # If there are any figures
            plt.pause(0.1)


if __name__ == '__main__':
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

        # Process files with non-blocking plots
        process_files_with_plots(selected_files)

        # Keep the program running so plots stay open
        print(f"\n✅ Processing complete! {len(selected_files)} plots created.")
        print("📊 Plots are now available. Close plot windows manually when done.")

        # Keep main thread alive to maintain plots
        try:
            while plt.get_fignums():  # While there are still open figures
                plt.pause(1)
        except KeyboardInterrupt:
            print("\n⏹️  Program interrupted by user.")
    else:
        print("No files were selected.")
