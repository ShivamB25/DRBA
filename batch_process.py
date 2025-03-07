import os
import argparse
import concurrent.futures
import time
import torch
import glob
from pathlib import Path
from tqdm import tqdm
import sys
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process videos with DRBA using multiple GPUs')
    parser.add_argument('-i', '--input_dir', dest='input_dir', type=str, required=True,
                        help='directory containing input video files')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default='output',
                        help='directory to save output video files')
    parser.add_argument('-m', '--model_type', dest='model_type', type=str, default='rife',
                        help='model network type, current support rife/gmfss/gmfss_union')
    parser.add_argument('-fps', '--dst_fps', dest='dst_fps', type=float, default=60, 
                        help='interpolate to ? fps')
    parser.add_argument('-t', '--times', dest='times', type=int, default=-1, 
                        help='interpolate to ?x fps')
    parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=False,
                        help='enable scene change detection')
    parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                        help='ssim scene detection threshold')
    parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=False,
                        help='enable hardware acceleration encode(require nvidia graph card)')
    parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                        help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
    parser.add_argument('-g', '--gpus', dest='gpus', type=str, default='all',
                        help='GPU IDs to use (comma-separated, e.g., "0,1,2" or "all" for all available GPUs)')
    parser.add_argument('-ext', '--extensions', dest='extensions', type=str, default='mp4,mkv,avi,mov',
                        help='File extensions to process (comma-separated)')
    parser.add_argument('-f', '--force', dest='force_overwrite', action='store_true', default=False,
                        help='Force overwrite existing output files')
    parser.add_argument('-j', '--jobs', dest='max_workers', type=int, default=0,
                        help='Maximum number of concurrent processes (default: number of GPUs)')
    return parser.parse_args()

def get_available_gpus():
    """Get the number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return []
    
    return list(range(torch.cuda.device_count()))

def load_model(model_type, scale, device):
    """Load the model based on model_type."""
    if model_type == 'rife':
        from models.rife import RIFE
        model = RIFE(weights=r'weights/train_log_rife_426_heavy', scale=scale, device=device)
    elif model_type == 'gmfss':
        from models.gmfss import GMFSS
        model = GMFSS(weights=r'weights/train_log_gmfss', scale=scale, device=device)
    elif model_type == 'gmfss_union':
        from models.gmfss_union import GMFSS_UNION
        model = GMFSS_UNION(weights=r'weights/train_log_gmfss_union', scale=scale, device=device)
    else:
        raise ValueError(f'model_type must be one of: rife, gmfss, gmfss_union')
    
    return model

def process_file(file_path, output_path, gpu_id, args):
    """Process a single file using the specified GPU."""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} does not exist")
        return False, file_path, gpu_id
    
    # Check if output file already exists and skip if it does (unless overwrite is enabled)
    if os.path.exists(output_path) and not args.force_overwrite:
        print(f"Output file {output_path} already exists, skipping (use -f to force overwrite)")
        return True, file_path, gpu_id
    
    # Create a log file for this process
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{os.path.basename(file_path)}.log")
    
    try:
        # Set the device and optimize CUDA settings
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            
            # Optimize CUDA for better performance
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            
            # Clear CUDA cache to ensure we have maximum available memory
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
        
        # Import necessary modules
        from models.utils.tools import VideoFI_IO, to_inp, to_out, get_valid_net_inp_size, TMapper, check_scene
        import numpy as np
        
        # Load the model
        model = load_model(args.model_type, args.scale, device)
        scale = args.scale
        dst_fps = args.dst_fps
        times = args.times
        enable_scdet = args.enable_scdet
        scdet_threshold = args.scdet_threshold
        hwaccel = args.hwaccel
        
        with open(log_file, 'w') as f:
            f.write(f"Processing {file_path} on GPU {gpu_id}\n")
            
            # Start inference
            video_io = VideoFI_IO(file_path, output_path, dst_fps=dst_fps, times=times, hwaccel=hwaccel)
            src_fps = video_io.src_fps
            
            if dst_fps <= src_fps:
                error_msg = f'dst fps should be greater than src fps, but got dst_fps={dst_fps} and src_fps={src_fps}'
                f.write(f"Error: {error_msg}\n")
                return False, file_path, gpu_id
            
            f.write(f"Source FPS: {src_fps}, Target FPS: {dst_fps}\n")
            f.write(f"Total frames: {video_io.total_frames_count}\n")
            
            # Create a progress bar
            pbar = tqdm(total=video_io.total_frames_count, desc=f"GPU {gpu_id}: {os.path.basename(file_path)}")
            
            # Start inference
            i0, i1 = video_io.read_frame(), video_io.read_frame()
            size = get_valid_net_inp_size(i0, model.scale, div=model.pad_size)
            src_size, dst_size = size['src_size'], size['dst_size']
            
            I0 = to_inp(i0, dst_size)
            I1 = to_inp(i1, dst_size)
            
            t_mapper = TMapper(src_fps, dst_fps, times)
            idx = 0
            
            def calc_t(_idx: float):
                if times != -1:
                    if times % 2:
                        vfi_timestamp = [(_i + 1) / times for _i in range((times - 1) // 2)]  # 0 ~ 0.5
                        vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [1] + [t + 1 for t in vfi_timestamp]
                        return np.array(vfi_timestamp)
                    else:
                        vfi_timestamp = [(_i + 0.5) / times for _i in range(times // 2)]  # 0 ~ 0.5
                        vfi_timestamp = list(reversed([1 - t for t in vfi_timestamp])) + [t + 1 for t in vfi_timestamp]
                        return np.array(vfi_timestamp)
                
                timestamp = np.array(
                    t_mapper.get_range_timestamps(_idx - 0.5, _idx + 0.5, lclose=True, rclose=False, normalize=False))
                vfi_timestamp = np.round(timestamp - _idx, 4) + 1  # [0.5, 1.5)
                
                return vfi_timestamp
            
            # head
            ts = calc_t(idx)
            left_scene = check_scene(I0, I1, scdet_threshold) if enable_scdet else False
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
                right_scene = check_scene(I1, I2, scdet_threshold) if enable_scdet else False
                
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
            
            f.write(f"\nCompleted processing {file_path} on GPU {gpu_id}\n")
            return True, file_path, gpu_id
                
    except Exception as e:
        import traceback
        with open(log_file, 'a') as f:
            f.write(f"\nException occurred: {str(e)}\n")
            f.write(traceback.format_exc())
        return False, file_path, gpu_id

def process_worker(file_info):
    """Worker function for multiprocessing."""
    file_path, output_path, gpu_id, args = file_info
    return process_file(file_path, output_path, gpu_id, args)

def main():
    # Initialize PyTorch multiprocessing
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # Get available GPUs
    if args.gpus.lower() == 'all':
        gpu_ids = get_available_gpus()
    else:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
    
    if not gpu_ids:
        print("No GPUs available. Using CPU.")
        gpu_ids = [-1]  # Use -1 to indicate CPU
    else:
        print(f"Using GPUs: {gpu_ids}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of video files in input directory
    extensions = args.extensions.split(',')
    video_files = []
    for ext in extensions:
        pattern = os.path.join(args.input_dir, f"*.{ext}")
        video_files.extend(glob.glob(pattern))
    
    if not video_files:
        print(f"No video files found in {args.input_dir} with extensions: {extensions}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Determine the number of worker processes
    max_workers = min(args.max_workers if args.max_workers > 0 else len(gpu_ids), len(video_files))
    print(f"Using {max_workers} concurrent worker processes")
    
    # Prepare tasks
    tasks = []
    for i, file_path in enumerate(video_files):
        # Assign GPU in round-robin fashion
        gpu_id = gpu_ids[i % len(gpu_ids)]
        
        # Create output path
        filename = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(filename)
        
        # Create a unique output path to avoid overwriting files with the same name
        output_path = os.path.join(args.output_dir, f"{file_base}{file_ext}")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add task
        tasks.append((file_path, output_path, gpu_id, args))
    
    # Process files in parallel using PyTorch multiprocessing
    completed = []
    failed = []
    
    # Create a progress bar for the main process
    pbar = tqdm(total=len(video_files), desc="Processing videos", unit="file")
    
    # Use a process pool to handle the tasks
    with mp.Pool(processes=max_workers) as pool:
        for result in pool.imap_unordered(process_worker, tasks):
            success, file_path, gpu_id = result
            if success:
                completed.append(file_path)
                pbar.set_postfix({"GPU": gpu_id, "Status": "Success"})
            else:
                failed.append(file_path)
                pbar.set_postfix({"GPU": gpu_id, "Status": "Failed"})
            pbar.update(1)
    
    pbar.close()
    
    # Print summary
    print(f"\nProcessing complete: {len(completed)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Batch processing completed in {elapsed_time:.2f} seconds")