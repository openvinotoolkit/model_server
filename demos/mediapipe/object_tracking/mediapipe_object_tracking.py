
import sys
import numpy as np
import cv2
import datetime
import argparse
import os
import tritonclient.grpc as grpcclient
import traceback
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description='Sends video frames via KServe gRPC API to OVMS MediaPipe graph and writes output video.'
    )
    parser.add_argument('--input_video', required=True,
                        help='Path to input MP4 video file.')
    parser.add_argument('--output_video', required=False, default='output.mp4',
                        help='Path to output MP4 video file. default: output.mp4')
    parser.add_argument('--grpc_address', required=False, default='localhost',
                        help='Specify url to grpc service. default: localhost')
    parser.add_argument('--grpc_port', required=False, default=9000, type=int,
                        help='Specify port to grpc service. default: 9000')
    parser.add_argument('--input_name', required=False, default='input_video',
                        help='Specify input tensor name. default: input_video')
    parser.add_argument('--output_name', required=False, default='output_video',
                        help='Specify output tensor name. default: output_video')
    parser.add_argument('--graph_name', required=False, default='objectTracking',
                        help='MediaPipe graph name as configured in OVMS. default: objectTracking')
    parser.add_argument('--tls', default=False, action='store_true',
                        help='Use TLS communication with gRPC endpoint.')
    parser.add_argument('--max_frames', required=False, default=0, type=int,
                        help='Max number of frames to process. 0 means all frames. default: 0')
    return vars(parser.parse_args())
 
 
def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, width, height, total_frames
 
 
def create_writer(path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video file: {path}")
        sys.exit(1)
    return writer
 
 
def create_grpc_client(address: str, tls: bool):
    try:
        client = grpcclient.InferenceServerClient(
            url=address,
            ssl=tls,
            verbose=False
        )
        return client
    except Exception as e:
        print(f"ERROR: Failed to create gRPC client: {e}")
        sys.exit(1)
 
 
def infer_frame(client, frame: np.ndarray, graph_name: str,
                input_name: str, output_name: str) -> np.ndarray:
    """Send a single frame to OVMS and return the output frame."""
    inputs = [grpcclient.InferInput(input_name, frame.shape, "UINT8")]
    inputs[0].set_data_from_numpy(np.array(frame, dtype=np.uint8))
 
    outputs = [grpcclient.InferRequestedOutput(output_name)]
 
    results = client.infer(
        model_name=graph_name,
        inputs=inputs,
        outputs=outputs
    )
    return results.as_numpy(output_name)
 
 
def main():
    args = parse_args()
 
    # Validate input file
    if not os.path.exists(args['input_video']):
        print(f"ERROR: Input video not found: {args['input_video']}")
        sys.exit(1)
 
    address = f"{args['grpc_address']}:{args['grpc_port']}"
    print(f"Connecting to OVMS at {address}")
    print(f"Graph name: {args['graph_name']}")
    print(f"Input video: {args['input_video']}")
    print(f"Output video: {args['output_video']}")
 
    # Open input video
    cap, fps, width, height, total_frames = open_video(args['input_video'])
    print(f"Video info: {width}x{height} @ {fps:.2f} fps, {total_frames} frames total")
 
    # Create gRPC client
    client = create_grpc_client(address, args['tls'])
 
    # Prepare output writer (size may change after first inference if graph resizes)
    writer = None
    output_path = args['output_video']
 
    processing_times = []
    frame_idx = 0
    max_frames = args['max_frames']
 
    print("\nProcessing frames...")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break
 
        start_time = datetime.datetime.now()
        try:
            output_frame = infer_frame(
                client, frame,
                args['graph_name'],
                args['input_name'],
                args['output_name']
            )
        except Exception as e:
            print(f"\nERROR on frame {frame_idx}: {e}")
            traces = traceback.extract_tb(e.__traceback__)
            print(traces)
            break
 
        end_time = datetime.datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        processing_times.append(duration_ms)
 
        # Initialise writer using actual output frame dimensions
        if writer is None:
            out_h, out_w = output_frame.shape[:2]
            writer = create_writer(output_path, fps, out_w, out_h)
            print(f"Output frame size: {out_w}x{out_h}")
 
        # Convert RGBA -> BGR if needed
        if output_frame.ndim == 3 and output_frame.shape[2] == 4:
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGBA2BGR)
        elif output_frame.ndim == 3 and output_frame.shape[2] == 3:
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
 
        writer.write(output_frame.astype(np.uint8))
 
        # Progress log every 10 frames
        if frame_idx % 10 == 0:
            avg_ms = np.mean(processing_times[-10:])
            print(f"  Frame {frame_idx:5d}/{total_frames} | "
                  f"{duration_ms:.1f} ms | "
                  f"{1000/avg_ms:.1f} fps avg",
                  end='\r')
 
        frame_idx += 1
 
    # Cleanup
    cap.release()
    if writer:
        writer.release()
 
    if processing_times:
        avg_ms = np.mean(processing_times)
        print(f"\n\nDone! Processed {frame_idx} frames.")
        print(f"Average processing time : {avg_ms:.2f} ms/frame")
        print(f"Average throughput      : {1000/avg_ms:.2f} fps")
        print(f"Output saved to         : {output_path}")
    else:
        print("\nNo frames were processed.")
 
 
if __name__ == '__main__':
    main()
 