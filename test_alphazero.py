import os
import shutil
import subprocess
import time
import multiprocessing as mp
import multiprocessing.connection 
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime as ort

from helper import BOARD_SIZE, SHAPES
from tf_alphazero_bot import ExpertBlokusBot
from player import BotPlayer  

def convert_unified_to_onnx(keras_path, onnx_path):
    if os.path.exists(onnx_path): return
    print(f"🚀 CONVERTING UNIFIED MODEL TO ONNX")
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)
    model.export("temp_savedmodel")
    subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", "temp_savedmodel", "--output", onnx_path, "--opset", "15"], check=True)
    if os.path.exists("temp_savedmodel"): shutil.rmtree("temp_savedmodel")

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
                # FIX: Check type to prevent array broadcast failure
                if isinstance(msg, str) and msg == "DONE": 
                    active_conns.remove(conn)
                else:
                    inputs = msg
                    length = len(inputs)
                    
                    if cursor + length > FIXED_BATCH:
                        if cursor > 0:
                            preds = fast_infer(buffer)
                            idx = 0
                            for c, l in zip(pipes, lens):
                                c.send((preds[0][idx:idx+l], preds[1][idx:idx+l], preds[2][idx:idx+l], preds[3][idx:idx+l]))
                                idx += l
                            pipes, lens, cursor = [], [], 0
                        
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
    adv_bot = ExpertBlokusBot(pipe=conn, is_training=False)
    az_wins, std_wins, ties, az_score_diff = 0, 0, 0, 0
    
    for _ in range(num_games):
        std_bot_1 = BotPlayer() 
        std_bot_2 = BotPlayer()
        
        board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        inventories = { 1: list(SHAPES.keys()), 2: list(SHAPES.keys()), 3: list(SHAPES.keys()), 4: list(SHAPES.keys()) }
        first_moves = {1: True, 2: True, 3: True, 4: True}
        
        if play_as_first:
            color_map = {1: adv_bot, 2: std_bot_1, 3: adv_bot, 4: std_bot_2}
        else:
            color_map = {1: std_bot_1, 2: adv_bot, 3: std_bot_2, 4: adv_bot}
            
        current_color = 1
        pass_count = 0

        while pass_count < 4:
            active_bot = color_map[current_color]
            
            if isinstance(active_bot, ExpertBlokusBot):
                shape_name, coords = active_bot.get_play(board, current_color, inventories, first_moves, pass_count)
            else:
                inv_key = str(current_color) if str(current_color) in inventories else current_color
                shape_name, coords = active_bot.get_play(board, current_color, inventories[inv_key], first_moves[current_color])
            
            if shape_name is None: pass_count += 1
            else:
                for r, c in coords: board[r][c] = current_color
                inventories[current_color].remove(shape_name)
                first_moves[current_color] = False
                pass_count = 0
                
            current_color = (current_color % 4) + 1

        def get_score(col_id): return sum(-int(shape.split('_')[0]) for shape in inventories[col_id])

        if play_as_first:
            az_score = get_score(1) + get_score(3)
            std_score = get_score(2) + get_score(4)
        else:
            std_score = get_score(1) + get_score(3)
            az_score = get_score(2) + get_score(4)

        az_score_diff += (az_score - std_score)
        if az_score > std_score: az_wins += 1
        elif std_score > az_score: std_wins += 1
        else: ties += 1
            
    return az_wins, std_wins, ties, az_score_diff

def distributed_test_worker(num_games, conns, result_queue, worker_idx):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    threads_count = len(conns)
    base_games = num_games // threads_count
    tasks = [base_games + (1 if i < num_games % threads_count else 0) for i in range(threads_count)]

    results = []
    with ThreadPoolExecutor(max_workers=threads_count) as executor:
        futures = [executor.submit(_thread_test_games, tasks[i], conns[i], i%2==0) for i in range(threads_count) if tasks[i]>0]
        for f in futures: results.append(f.result())

    for c in conns: c.send("DONE") 
    result_queue.put(np.sum(results, axis=0).tolist())

def test_model(num_games=100, threads_per_worker=10):
    print("="*60)
    print(f"DISTRIBUTED TENSORRT BENCHMARK (BLOKUS - {num_games} GAMES)")
    print("="*60)
    
    keras_path = "blokus_expert_v0.keras"
    onnx_path = "blokus_expert.onnx"
    
    if not os.path.exists(keras_path) and not os.path.exists(onnx_path):
        print(f"❌ Waiting for train_alphazero.py to output {keras_path} first.")
        return
        
    convert_unified_to_onnx(keras_path, onnx_path)

    # UNLEASHED 8GB MEMORY
    trt_providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_max_workspace_size': 8589934592  
        }),
        'CUDAExecutionProvider'
    ]

    print("Warming up TRT Engine across 8GB Workspace...")
    session = ort.InferenceSession(onnx_path, providers=trt_providers)
    in_name = session.get_inputs()[0].name
    
    def fast_infer(batch_np): 
        return session.run(None, {in_name: batch_np.astype(np.float32)})

    _ = fast_infer(np.zeros((4096, 20, 20, 6), dtype=np.float16))
    
    num_workers = min(31, mp.cpu_count() - 1) 
    print(f"Starting {num_workers} CPU Process Workers...")
    
    start_time = time.time()
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

    total_aw, total_sw, total_t, total_diff = 0, 0, 0, 0
    for _ in range(len(processes)):
        aw, sw, t, diff = result_queue.get()
        total_aw += aw; total_sw += sw; total_t += t; total_diff += diff

    for p in processes: p.join()
    elapsed = time.time() - start_time
    avg_diff = total_diff / num_games

    print("\n" + "="*60)
    print(f"FINAL BENCHMARK RESULTS (Completed in {elapsed:.1f}s)")
    print("="*60)
    print(f"Expert Bot Wins: {total_aw} ({(total_aw/num_games)*100:.1f}%)")
    print(f"BotPlayer Wins:  {total_sw} ({(total_sw/num_games)*100:.1f}%)")
    print(f"Ties:            {total_t} ({(total_t/num_games)*100:.1f}%)")
    print(f"Avg Point Diff:  {'+' if avg_diff > 0 else ''}{avg_diff:.1f} pts per game for Expert Bot")
    print("="*60)

if __name__ == "__main__":
    mp.freeze_support()
    test_model(num_games=200, threads_per_worker=10)