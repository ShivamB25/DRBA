# DRBA Batch Processing

This tool allows you to batch process multiple video files using DRBA (Deep Recursive Bilateral Adjustment) with multiple GPUs.

## Features

- Process multiple video files in parallel
- Distribute processing across multiple GPUs
- Automatically skip already processed files
- Detailed logging for each processed file
- Progress tracking with ETA
- Customizable output directory

## Requirements

- Python 3.6+
- PyTorch with CUDA support
- DRBA dependencies (as per the main project)
- Multiple NVIDIA GPUs (optional, but recommended for faster processing)

## Usage

```bash
python batch_process.py -i INPUT_DIRECTORY -o OUTPUT_DIRECTORY [options]
```

### Basic Examples

Process all video files in the "videos" directory and save results to "output" directory:
```bash
python batch_process.py -i videos -o output
```

Process all video files using the GMFSS model:
```bash
python batch_process.py -i videos -o output -m gmfss
```

Process all video files using specific GPUs (0 and 1):
```bash
python batch_process.py -i videos -o output -g 0,1
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input_dir` | Directory containing input video files | (Required) |
| `-o, --output_dir` | Directory to save output video files | "output" |
| `-m, --model_type` | Model network type (rife/gmfss/gmfss_union) | "rife" |
| `-fps, --dst_fps` | Target FPS for interpolation | 60 |
| `-t, --times` | Interpolate to X times the original FPS | -1 (use dst_fps) |
| `-s, --enable_scdet` | Enable scene change detection | False |
| `-st, --scdet_threshold` | SSIM scene detection threshold | 0.3 |
| `-hw, --hwaccel` | Enable hardware acceleration for encoding | False |
| `-scale, --scale` | Flow scale (1.0 for 1080p, 0.5 for 4K) | 1.0 |
| `-g, --gpus` | GPU IDs to use (comma-separated or "all") | "all" |
| `-ext, --extensions` | File extensions to process (comma-separated) | "mp4,mkv,avi,mov" |
| `-f, --force` | Force overwrite existing output files | False |
| `-j, --jobs` | Maximum number of concurrent processes | (Number of GPUs) |

## Output Structure

- Processed videos are saved to the specified output directory
- Log files are saved in the `output/logs` directory (one log file per processed video)

## Tips

1. For 4K videos, use `-scale 0.5` for better performance
2. If you have limited VRAM, reduce the number of concurrent jobs with `-j`
3. Use `-f` to force overwrite existing output files
4. For videos with scene changes, enable scene detection with `-s`