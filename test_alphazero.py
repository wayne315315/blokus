import os
import shutil
import subprocess
import time
import multiprocessing as mp
import multiprocessing.connection 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime as ort

from helper import BOARD_SIZE
from tf_alphazero_bot import ExpertBlokusBot
from player import BotPlayer  

def convert_unified_to_onnx(keras_path, onnx_path):
    if os.path.exists(onnx_path): return
    print(f"🚀 CONVERTING UNIFIED MODEL TO ONNX")
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)
    model.export("temp_savedmodel")
    subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", "temp_savedmodel", "--output", onnx_path, "--opset", "15"], check=True)
    shutil.rmtree("temp_savedmodel")

def unified_inference_server(conns, fast_infer):
    FIXED_BATCH = 4096 
    active_conns = list(conns)
    buffer = np.zeros((FIXED_BATCH, 20, 20, 6), dtype=np.float16)
    pipes, lens = [], []
    cursor, last_fire = 0, time.time()

    while active_conns:
        ready = multiprocessing.connection.wait(active_conns, timeout=0.01)
        for conn in ready:
            try:
                msg = conn.recv()
                if msg == "DONE": active_conns.remove(conn)
                else:
                    inputs = msg
                    length = len(inputs)
                    
                    # Split massive requests to prevent ValueError buffer crashes
                    if cursor + length > FIXED_BATCH:
                        if cursor > 0:
                            preds = fast_infer(buffer)
                            idx = 0
                            for c, l in zip(pipes, lens):
                                c.send((preds[0][idx:idx+l], preds[1][idx:idx+l], preds[2][idx:idx+l], preds[3][idx:idx+l]))
                                idx += l
                            pipes, lens, cursor = [], [], 0
                        
                        # Process overflow request dynamically
                        if length > FIXED_BATCH:
                            big_preds = fast_infer(inputs)
                            conn.send(big_preds)
                            continue
                            
                    buffer[cursor : cursor + length] = inputs
                    pipes.append(conn)
                    lens.append(length)
                    cursor += length
            except EOFError:
                if conn in active_conns: active_conns.remove(conn)

        if cursor >= 128 or (time.time() - last_fire > 0.05 and cursor > 0):
            preds = fast_infer(buffer)
            idx = 0
            for c, l in zip(pipes, lens):
                c.send((preds[0][idx:idx+l], preds[1][idx:idx+l], preds[2][idx:idx+l], preds[3][idx:idx+l]))
                idx += l
            pipes, lens, cursor, last_fire = [], [], 0, time.time()

def _thread_test_games(num_games, conn, play_as_first):
    # Initializes UI-compatible bot
    adv_bot = ExpertBlokusBot(pipe=conn, is_training=False)
    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    
    # ... Play loop identical to previous iterations...
    return az_wins, std_wins, ties, az_score_diff

def distributed_test_worker(num_games, conns, result_queue, worker_idx):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Hide GPU from CPU workers
    threads_count = len(conns)
    base_games = num_games // threads_count
    tasks = [base_games + (1 if i < num_games % threads_count else 0) for i in range(threads_count)]

    results = []
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = [executor.submit(_thread_test_games, tasks[i], conns[i], i%2==0) for i in range(threads_count) if tasks[i]>0]
        for f in futures: results.append(f.result())

    # Aggregate & Clean up...
    for c in conns: c.send("DONE") 
    result_queue.put(np.sum(results, axis=0).tolist())

def test_model(num_games=100, threads_per_worker=10):
    convert_unified_to_onnx("blokus_expert_v0.keras", "blokus_expert.onnx")

    # Explicitly instruct TensorRT to use up to 8GB VRAM (No more 2GB limits)
    trt_providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_max_workspace_size': 8589934592  # 8GB Full VRAM allocation
        }),
        'CUDAExecutionProvider'
    ]

    print("Warming up TRT Engine across 8GB Workspace...")
    session = ort.InferenceSession("blokus_expert.onnx", providers=trt_providers)
    in_name = session.get_inputs()[0].name
    
    def fast_infer(batch_np): 
        return session.run(None, {in_name: batch_np.astype(np.float32)})

    # Warmup
    _ = fast_infer(np.zeros((4096, 20, 20, 6), dtype=np.float16))
    
    num_workers = min(15, mp.cpu_count() - 1) # Utilize the Ryzen 9 3950X
    print(f"Starting {num_workers} CPU Workers...")
    
    ctx = mp.get_context('spawn')
    pipes = [ctx.Pipe() for _ in range(num_workers * threads_per_worker)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
    
    result_queue = ctx.Queue()
    processes = []
    
    for i in range(num_workers):
        p = ctx.Process(target=distributed_test_worker, args=(num_games//num_workers, child_conns[i*threads_per_worker:(i+1)*threads_per_worker], result_queue, i))
        p.start()
        processes.append(p)

    unified_inference_server(parent_conns, fast_infer)

    for p in processes: p.join()
    print("Benchmark complete!")

if __name__ == "__main__":
    mp.freeze_support()
    test_model(num_games=100)
