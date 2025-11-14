import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.dag_task_node import DAGTask

# Import centralized configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.config import WORKER_CONFIG

def create_matrix_chunks(matrix, row_chunk_size=1, col_chunk_size=1):
    """Split matrix into smaller chunks based on specified sizes"""
    chunks = []
    for i in range(0, matrix.shape[0], row_chunk_size):
        for j in range(0, matrix.shape[1], col_chunk_size):
            chunk = matrix[i:i+row_chunk_size, j:j+col_chunk_size]
            chunks.append(((i, j), chunk))  # Store position and chunk
    return chunks

@DAGTask
def multiply_chunks(a_chunk_with_pos, b_chunk_with_pos):
    (i_a, _), a_chunk = a_chunk_with_pos
    (_, j_b), b_chunk = b_chunk_with_pos

    product = np.matmul(a_chunk, b_chunk)

    return ((i_a, j_b), product)


@DAGTask
def aggregate_results(partial_results, final_shape):
    result = np.zeros(final_shape)
    for position, value in partial_results:
        i, j = position
        rows, cols = value.shape
        result[i:i+rows, j:j+cols] = value

    return result


RANDOM_MATRIX_COLS = 2_000
RANDOM_MATRIX_ROWS = 2_000
CHUNK_SIZE = 500

def generate_matrices(rows_a, cols_a):
    matrix_a = np.random.randint(1, 10, (rows_a, cols_a))
    matrix_b = np.random.randint(1, 10, (cols_a, rows_a))
    
    return matrix_a, matrix_b

start_time = time.time()
matrix_a, matrix_b = generate_matrices(RANDOM_MATRIX_ROWS, RANDOM_MATRIX_COLS)
print(f"Random matrices ({RANDOM_MATRIX_ROWS}x{RANDOM_MATRIX_COLS}) generated in {time.time() - start_time:.4f} seconds")

start_time = time.time()
# ! Not included in the workflow, not @DAGTask
a_chunks = create_matrix_chunks(matrix_a, row_chunk_size=CHUNK_SIZE, col_chunk_size=matrix_a.shape[1])
# ! Not included in the workflow, not @DAGTask
b_chunks = create_matrix_chunks(matrix_b, row_chunk_size=matrix_b.shape[0], col_chunk_size=CHUNK_SIZE)
print(f"Created {len(a_chunks) + len(b_chunks)} chunks for matrices in {time.time() - start_time:.4f} seconds")

start_time = time.time()
partial_results = []
for a_chunk in a_chunks:
    for b_chunk in b_chunks:
        result = multiply_chunks(a_chunk, b_chunk)
        partial_results.append(result)

print(f"Created {len(partial_results)} partial results in {time.time() - start_time:.4f} seconds")

result = aggregate_results(partial_results, (matrix_a.shape[0], matrix_b.shape[1]))

# result.visualize_dag(output_file=os.path.join("_dag_visualization", "gemm"), open_after=False)
# exit()

start_time = time.time()
result = result.compute(dag_name="gemm", config=WORKER_CONFIG, open_dashboard=False)
print(f"User waited: {time.time() - start_time:.2f}s")
# print(f"Is Multiplication correct: {np.allclose(np.matmul(matrix_a, matrix_b), result)}")