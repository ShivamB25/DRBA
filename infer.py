import os
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
import time
import tempfile
import shutil
import subprocess
from models.utils.tools import *
import warnings

warnings.filterwarnings("ignore")

# Check available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    device = torch.device("cpu")
    print("No CUDA devices available, using CPU")
else:
    device = torch.device("cuda")
    print(f"Found {num_gpus} CUDA devices")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Interpolation a video with DRBA')
    parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='rife',
                        help='model network type, current support rife/gmfss/gmfss_union')
    parser.add_argument('-i', '--input', dest='input', type=str, default='input.mp4',
                        help='absolute path of input video')
    parser.add_argument('-o', '--output', dest='output', type=str, default='output.mp4',
                        help='absolute path of output video')
    parser.add_argument('-fps', '--dst_fps', dest='dst_fps', type=float, default=60, help='interpolate to ? fps')
    parser.add_argument('-t', '--times', dest='times', type=int, default=-1, help='interpolate to ?x fps')
    parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                        help='enable scene change detection')
    parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                        help='ssim scene detection threshold')
    parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=False,
                        help='enable hardware acceleration encode(require nvidia graph card)')
    parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                        help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
    parser.add_argument('-c', '--chunk_size', dest='chunk_size', type=int, default=None,
                        help='number of frames per chunk (default: auto-calculated based on video length and GPU count)')
    return parser.parse_args()


def load_model(model_type, model_scale=1.0):
    """Load the model on the default CUDA device"""
    if model_type == 'rife':
        from models.rife import RIFE
        model = RIFE(weights=r'weights/train_log_rife_426_heavy', scale=model_scale, device=device)
    elif model_type == 'gmfss':
        from models.gmfss import GMFSS
        model = GMFSS(weights=r'weights/train_log_gmfss', scale=model_scale, device=device)
    elif model_type == 'gmfss_union':
        from models.gmfss_union import GMFSS_UNION
        model = GMFSS_UNION(weights=r'weights/train_log_gmfss_union', scale=model_scale, device=device)
    else:
        raise ValueError(f'model_type must in {model_type}')
    return model


def process_chunk(chunk_info):
    """Process a specific chunk of the video"""
    chunk_id, start_frame, end_frame, input_path, output_path, model_type, gpu_id, args = chunk_info
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"Chunk {chunk_id}: Using GPU {gpu_id}")
    
    # Load model
    local_model = load_model(model_type, args.scale)
    
    # Create a video reader for the input
    video_capture = cv2.VideoCapture(input_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Skip to the start frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create a video writer for the output
    video_io = VideoFI_IO(input_path, output_path, dst_fps=args.dst_fps, times=args.times, hwaccel=args.hwaccel)
    
    # Calculate the number of frames to process
    frames_to_process = min(end_frame - start_frame, total_frames - start_frame)
    pbar = tqdm(total=frames_to_process, desc=f"Chunk {chunk_id} (GPU {gpu_id})")
    
    # Read the first two frames
    ret, i0 = video_capture.read()
    if not ret:
        print(f"Chunk {chunk_id}: Failed to read first frame")
        return None
    
    ret, i1 = video_capture.read()
    if not ret:
        print(f"Chunk {chunk_id}: Failed to read second frame")
        # Write the first frame and return
        video_io.write_frame(i0)
        while not video_io.finish_writing():
            time.sleep(1)
        return output_path
    
    size = get_valid_net_inp_size(i0, local_model.scale, div=local_model.pad_size)
    src_size, dst_size = size['src_size'], size['dst_size']
    
    I0 = to_inp(i0, dst_size)
    I1 = to_inp(i1, dst_size)
    
    t_mapper = TMapper(src_fps, args.dst_fps, args.times)
    idx = 0
    frames_processed = 0
    
    def calc_t(_idx: float):
        if args.times != -1:
            if args.times % 2:
                vfi_timestamp = [(_i + 1) / args.times for _i in range((args.times - 1) // 2)]  # 0 ~ 0.5
                vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [1] + [t + 1 for t in vfi_timestamp]
                return np.array(vfi_timestamp)
            else:
                vfi_timestamp = [(_i + 0.5) / args.times for _i in range(args.times // 2)]  # 0 ~ 0.5
                vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [t + 1 for t in vfi_timestamp]
                return np.array(vfi_timestamp)

        timestamp = np.array(
            t_mapper.get_range_timestamps(_idx - 0.5, _idx + 0.5, lclose=True, rclose=False, normalize=False))
        vfi_timestamp = np.round(timestamp - _idx, 4) + 1  # [0.5, 1.5)

        return vfi_timestamp
    
    # Process frames
    try:
        # head
        ts = calc_t(idx)
        left_scene = check_scene(I0, I1, args.scdet_threshold) if args.enable_scdet else False
        right_scene = left_scene
        reuse = None
        
        if right_scene:
            output = [I0 for _ in ts]
        else:
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1] - 1
            
            output = [I0 for _ in left_ts]
            output.extend(local_model.inference_ts(I0, I1, right_ts))
        
        for x in output:
            video_io.write_frame(to_out(x, src_size))
        pbar.update(1)
        frames_processed += 1
        
        # Process the remaining frames in the chunk
        while frames_processed < frames_to_process - 1:
            ret, i2 = video_capture.read()
            if not ret:
                break
            
            I2 = to_inp(i2, dst_size)
            
            ts = calc_t(idx)
            right_scene = check_scene(I1, I2, args.scdet_threshold) if args.enable_scdet else False
            
            # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
            if left_scene and right_scene:  # scene transition occurs at I0~I1, also occurs at I1~I2
                output = [I1 for _ in ts]
                reuse = None
                
            elif left_scene and not right_scene:  # scene transition occurs at I0~I1
                left_ts = ts[ts < 1]
                right_ts = ts[ts >= 1] - 1
                reuse = None
                
                output = [I1 for _ in left_ts]
                output.extend(local_model.inference_ts(I1, I2, right_ts))
                
            elif not left_scene and right_scene:  # scene transition occurs at I1~I2
                left_ts = ts[ts <= 1]
                right_ts = ts[ts > 1] - 1
                reuse = None
                
                output = local_model.inference_ts(I0, I1, left_ts)
                output.extend([I1 for _ in right_ts])
                
            else:  # no scene transition
                output, reuse = local_model.inference_ts_drba(I0, I1, I2, ts, reuse, linear=True)
            
            for x in output:
                video_io.write_frame(to_out(x, src_size))
            
            i0, i1 = i1, i2
            I0, I1 = I1, I2
            left_scene = right_scene
            idx += 1
            frames_processed += 1
            pbar.update(1)
        
        # tail
        if frames_processed < frames_to_process:
            ts = calc_t(idx)
            left_ts = ts[ts <= 1]
            right_ts = ts[ts > 1] - 1
            
            output = local_model.inference_ts(I0, I1, left_ts)
            output.extend([I1 for _ in right_ts])
            
            for x in output:
                video_io.write_frame(to_out(x, src_size))
            pbar.update(1)
        
        # wait for output
        while not video_io.finish_writing():
            time.sleep(1)
        
        print(f"Chunk {chunk_id} (GPU {gpu_id}) completed")
        return output_path
    
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {str(e)}")
        return None
    finally:
        pbar.close()
        video_capture.release()


def process_video_multi_gpu():
    """Process the video in parallel using all available GPUs"""
    # Get video information
    video_capture = cv2.VideoCapture(input_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    
    if args.dst_fps <= src_fps and args.times == -1:
        raise ValueError(f'dst fps should be greater than src fps, but got dst_fps={args.dst_fps} and src_fps={src_fps}')
    
    # Use all available GPUs
    available_gpus = list(range(num_gpus))
    num_workers = max(1, len(available_gpus))
    
    print(f"Using {num_workers} GPU{'s' if num_workers > 1 else ''}: {available_gpus}")
    
    # Determine chunk size
    chunk_size = args.chunk_size if args.chunk_size else max(10, total_frames // (num_workers * 2))
    overlap = 2  # Fixed overlap of 2 frames between chunks
    
    # Create temporary directory for chunk outputs
    temp_dir = tempfile.mkdtemp()
    try:
        # Split the video into chunks
        chunks = []
        for i in range(0, total_frames, chunk_size - overlap):
            start_frame = max(0, i)
            end_frame = min(total_frames, i + chunk_size)
            
            # Skip tiny chunks at the end
            if end_frame - start_frame < 5:
                continue
                
            chunk_output = os.path.join(temp_dir, f"chunk_{i}.mp4")
            gpu_id = len(chunks) % num_workers
            chunks.append((len(chunks), start_frame, end_frame, input_path, chunk_output, args.model_type, gpu_id, args))
        
        print(f"Split video into {len(chunks)} chunks")
        
        # Process chunks in parallel
        if num_workers > 1:
            # Initialize multiprocessing
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=num_workers) as pool:
                chunk_outputs = pool.map(process_chunk, chunks)
        else:
            # Process sequentially if only one worker
            chunk_outputs = [process_chunk(chunk) for chunk in chunks]
        
        # Combine chunks
        if len(chunks) > 1:
            print("Combining chunks...")
            # Create a file with the list of chunk files
            chunk_list_file = os.path.join(temp_dir, "chunks.txt")
            with open(chunk_list_file, 'w') as f:
                for chunk_output in chunk_outputs:
                    if chunk_output and os.path.exists(chunk_output):
                        f.write(f"file '{chunk_output}'\n")
            
            # Use ffmpeg to concatenate the chunks
            concat_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', chunk_list_file, '-c', 'copy', output_path
            ]
            subprocess.run(concat_cmd, check=True)
            print(f"Output saved to {output_path}")
        elif len(chunks) == 1 and chunk_outputs[0]:
            # Just copy the single chunk
            shutil.copy2(chunk_outputs[0], output_path)
            print(f"Output saved to {output_path}")
        else:
            print("Error: No chunks were processed successfully")
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def inference_single_gpu():
    """Legacy single-GPU inference function"""
    video_io = VideoFI_IO(input_path, output_path, dst_fps=args.dst_fps, times=args.times, hwaccel=args.hwaccel)
    src_fps = video_io.src_fps
    if args.dst_fps <= src_fps and args.times == -1:
        raise ValueError(f'dst fps should be greater than src fps, but got dst_fps={args.dst_fps} and src_fps={src_fps}')
    pbar = tqdm(total=video_io.total_frames_count)

    # start inference
    i0, i1 = video_io.read_frame(), video_io.read_frame()
    size = get_valid_net_inp_size(i0, model.scale, div=model.pad_size)
    src_size, dst_size = size['src_size'], size['dst_size']

    I0 = to_inp(i0, dst_size)
    I1 = to_inp(i1, dst_size)

    t_mapper = TMapper(src_fps, args.dst_fps, args.times)
    idx = 0

    def calc_t(_idx: float):
        if args.times != -1:
            if args.times % 2:
                vfi_timestamp = [(_i + 1) / args.times for _i in range((args.times - 1) // 2)]  # 0 ~ 0.5
                vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [1] + [t + 1 for t in vfi_timestamp]
                return np.array(vfi_timestamp)
            else:
                vfi_timestamp = [(_i + 0.5) / args.times for _i in range(args.times // 2)]  # 0 ~ 0.5
                vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [t + 1 for t in vfi_timestamp]
                return np.array(vfi_timestamp)

        timestamp = np.array(
            t_mapper.get_range_timestamps(_idx - 0.5, _idx + 0.5, lclose=True, rclose=False, normalize=False))
        vfi_timestamp = np.round(timestamp - _idx, 4) + 1  # [0.5, 1.5)

        return vfi_timestamp

    # head
    ts = calc_t(idx)
    left_scene = check_scene(I0, I1, args.scdet_threshold) if args.enable_scdet else False
    right_scene = left_scene
    reuse = None

    if right_scene:
        output = [I0 for _ in ts]
    else:
        left_ts = ts[ts < 1]
        right_ts = ts[ts >= 1] - 1

        output = [I0 for _ in left_ts]
        output.extend(model.inference_ts(I0, I1, right_ts))

    for x in output:
        video_io.write_frame(to_out(x, src_size))
    pbar.update(1)

    while True:
        i2 = video_io.read_frame()
        if i2 is None:
            break
        I2 = to_inp(i2, dst_size)

        ts = calc_t(idx)
        right_scene = check_scene(I1, I2, args.scdet_threshold) if args.enable_scdet else False

        # If a scene transition occurs between the three frames, then the calculation of this DRM is meaningless.
        if left_scene and right_scene:  # scene transition occurs at I0~I1, also occurs at I1~I2
            output = [I1 for _ in ts]
            reuse = None

        elif left_scene and not right_scene:  # scene transition occurs at I0~I1
            left_ts = ts[ts < 1]
            right_ts = ts[ts >= 1] - 1
            reuse = None

            output = [I1 for _ in left_ts]
            output.extend(model.inference_ts(I1, I2, right_ts))

        elif not left_scene and right_scene:  # scene transition occurs at I1~I2
            left_ts = ts[ts <= 1]
            right_ts = ts[ts > 1] - 1
            reuse = None

            output = model.inference_ts(I0, I1, left_ts)
            output.extend([I1 for _ in right_ts])

        else:  # no scene transition
            output, reuse = model.inference_ts_drba(I0, I1, I2, ts, reuse, linear=True)

        for x in output:
            video_io.write_frame(to_out(x, src_size))

        i0, i1 = i1, i2
        I0, I1 = I1, I2
        left_scene = right_scene
        idx += 1
        pbar.update(1)

    # tail
    ts = calc_t(idx)
    left_ts = ts[ts <= 1]
    right_ts = ts[ts > 1] - 1

    output = model.inference_ts(I0, I1, left_ts)
    output.extend([I1 for _ in right_ts])

    for x in output:
        video_io.write_frame(to_out(x, src_size))
    idx += 1
    pbar.update(1)

    # wait for output
    while not video_io.finish_writing():
        time.sleep(1)
    pbar.close()


if __name__ == '__main__':
    args = parse_args()
    input_path = args.input  # input video path
    output_path = args.output  # output video path

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"can't find the video file {input_path}")

    # Determine if we should use multi-GPU processing
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
        print(f"Using multi-GPU processing with {num_gpus} available GPUs")
        process_video_multi_gpu()
    else:
        print("Using single-GPU processing")
        model = load_model(args.model_type, args.scale)
        inference_single_gpu()
