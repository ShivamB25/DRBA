import os
import argparse
import subprocess
import concurrent.futures
import time
import torch
import glob
from pathlib import Path
from tqdm import tqdm
import sys

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
    
    # Build command with all necessary arguments
    cmd = [
        "python", "infer.py",
        "-m", args.model_type,
        "-i", file_path,
        "-o", output_path,
        "-fps", str(args.dst_fps),
        "-t", str(args.times),
        "-scale", str(args.scale),
        "-gpu", str(gpu_id)
    ]
    
    # Add optional flags
    if args.enable_scdet:
        cmd.append("-s")
    if args.hwaccel:
        cmd.append("-hw")
    if args.scdet_threshold != 0.3:  # Only add if not default
        cmd.extend(["-st", str(args.scdet_threshold)])
    
    # Create a log file for this process
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{os.path.basename(file_path)}.log")
    
    # Run the command
    try:
        with open(log_file, 'w') as f:
            f.write(f"Processing {file_path} on GPU {gpu_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output to log file
            for line in process.stdout:
                f.write(line)
                f.flush()
            
            return_code = process.wait()
            
            if return_code == 0:
                f.write(f"\nCompleted processing {file_path} on GPU {gpu_id}\n")
                return True, file_path, gpu_id
            else:
                f.write(f"\nError processing {file_path} on GPU {gpu_id}: Return code {return_code}\n")
                return False, file_path, gpu_id
                
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\nException occurred: {str(e)}\n")
        return False, file_path, gpu_id

def main():
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
    
    # Determine the number of worker threads
    max_workers = args.max_workers if args.max_workers > 0 else len(gpu_ids)
    print(f"Using {max_workers} concurrent worker threads")
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # Create a progress bar
        pbar = tqdm(total=len(video_files), desc="Processing videos", unit="file")
        
        # Submit tasks for each file
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
            
            # Submit task
            future = executor.submit(process_file, file_path, output_path, gpu_id, args)
            futures[future] = (file_path, gpu_id)
        
        # Track completed and failed files
        completed = []
        failed = []
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            file_path, gpu_id = futures[future]
            try:
                success, _, _ = future.result()
                if success:
                    completed.append(file_path)
                    pbar.set_postfix({"GPU": gpu_id, "Status": "Success"})
                else:
                    failed.append(file_path)
                    pbar.set_postfix({"GPU": gpu_id, "Status": "Failed"})
            except Exception as e:
                failed.append(file_path)
                print(f"\nUnexpected error processing {file_path} on GPU {gpu_id}: {e}", file=sys.stderr)
            
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