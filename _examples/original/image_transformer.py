import os
import sys
import time
import numpy as np
from PIL import Image, ImageFilter

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dag_task_node import DAGTask

# Import centralized configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.config import WORKER_CONFIG


# --- DAG Tasks ---

@DAGTask
def split_image(img: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return exactly 16 chunks (4x4 grid) in row-major order."""
    h, w = img.shape[:2]
    h_step, w_step = h // 4, w // 4
    chunks = []
    for i in range(4):
        for j in range(4):
            chunk = img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step].copy()
            chunks.append(chunk)
    time.sleep(9) # to influence planner to assign prewarm
    return tuple(chunks)  # single DAG node output (tuple)


@DAGTask
def get_chunk(chunks: tuple[np.ndarray, ...], idx: int) -> np.ndarray:
    """Extract a single chunk from the split output."""
    return chunks[idx]


@DAGTask
def resize_chunk(chunk: np.ndarray, size=None) -> np.ndarray:
    img = Image.fromarray(chunk)
    if size is None:
        size = (chunk.shape[1], chunk.shape[0])  # keep original chunk size
    img_resized = img.resize(size)
    return np.array(img_resized)


@DAGTask
def blur_chunk(chunk: np.ndarray) -> np.ndarray:
    img = Image.fromarray(chunk)
    img_blurred = img.filter(ImageFilter.GaussianBlur(1))
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 8192, dtype=np.float64))) # cpu-bound work
    return np.array(img_blurred)


@DAGTask
def sepia_chunk(chunk: np.ndarray) -> np.ndarray:
    """Apply sepia filter to an image chunk."""
    img = chunk.astype(np.float32)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = img @ sepia_filter.T
    sepia_img = np.clip(sepia_img, 0, 255)
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 8192, dtype=np.float64))) # cpu-bound work
    return sepia_img.astype(np.uint8)


@DAGTask
def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64))) # cpu-bound work
    return ((chunk - chunk.min()) / (chunk.max() - chunk.min()) * 255).astype(np.uint8)


@DAGTask
def edge_detect_chunk(chunk: np.ndarray) -> np.ndarray:
    img = Image.fromarray(chunk).convert("L")
    img_edge = img.filter(ImageFilter.FIND_EDGES)
    img_edge = img_edge.resize((chunk.shape[1], chunk.shape[0]))
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 8192, dtype=np.float64))) # cpu-bound work
    return np.array(img_edge)


@DAGTask
def sharpen_chunk(chunk: np.ndarray) -> np.ndarray:
    img = Image.fromarray(chunk)
    img_sharp = img.filter(ImageFilter.UnsharpMask())
    img_sharp = img_sharp.resize((chunk.shape[1], chunk.shape[0]))
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 8192, dtype=np.float64))) # cpu-bound work
    return np.array(img_sharp)


@DAGTask
def combine_chunk(branch_a: np.ndarray, branch_b: np.ndarray) -> np.ndarray:
    if branch_b.shape != branch_a.shape:
        if len(branch_b.shape) == 2:  # grayscale → 3 channels
            branch_b = np.stack([branch_b] * 3, axis=-1)
        branch_b = np.array(Image.fromarray(branch_b).resize((branch_a.shape[1], branch_a.shape[0])))
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64))) # cpu-bound work
    return ((branch_a.astype(np.float32) + branch_b.astype(np.float32)) / 2).astype(np.uint8)


@DAGTask
def merge_chunks_grid(chunks: list[np.ndarray], grid_size: int = 4) -> np.ndarray:
    """Combine 16 chunks back into a full image (4x4 grid)."""
    rows = []
    for r in range(grid_size):
        row_chunks = chunks[r * grid_size:(r + 1) * grid_size]
        row = np.hstack(row_chunks)
        rows.append(row)
    full_image = np.vstack(rows)
    _ = np.sum(np.sqrt(np.linspace(0.0, 1.0, 4096, dtype=np.float64))) # cpu-bound work
    return full_image


def _save_image(img: np.ndarray, path: str) -> str:
    Image.fromarray(img).save(path)
    return path


# --- Define Workflow ---

input_file = "../_inputs/test_image_2.jpg"
output_file = "../_outputs/image_transform_2.jpg"

img = np.array(Image.open(input_file))

# Split into 16 chunks (lazy reference to tuple)
split_out = split_image(img)

# Define per-chunk sub-pipeline
def process_chunk(split_out, idx: int):
    chunk = get_chunk(split_out, idx)

    a1 = resize_chunk(chunk)
    a2 = blur_chunk(a1)
    a3 = normalize_chunk(a2)
    a4 = sepia_chunk(a3)

    b1 = edge_detect_chunk(chunk)
    b2 = sharpen_chunk(b1)

    return combine_chunk(a4, b2)


# Apply statically to all 16 indices
processed_chunks = [
    process_chunk(split_out, i)
    for i in range(16)
]

# Merge final image
final_img = merge_chunks_grid(processed_chunks, grid_size=4)

# --- Run Workflow ---
# final_img.visualize_dag(output_file=os.path.join("_dag_visualization", "image_transformer"), open_after=True)

start_time = time.time()
result = final_img.compute(dag_name="image_transformer", config=WORKER_CONFIG, open_dashboard=False)
print(f"User waited: {time.time() - start_time:.3f}s")

# _save_image(result, output_file)
