import numpy as np
import soundfile as sf
from scipy.signal import hilbert
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

def aphex_equation(D, F, F_ext, alpha):
    n = F.shape[1]
    C = min(F.shape[0], D.shape[0])
    result = np.zeros_like(F_ext)
    
    for i in range(n):
        sum_F = np.sum([F[j][i] for j in range(C)])
        clipped_D = np.clip(D[0][i], -700, 700)
        result[i] = (sum_F + F_ext[i]) * np.exp(-alpha * clipped_D)
    
    return result

def process_audio_segment(audio_segment, sample_rate, alpha):
    processed_channels = []
    for channel in range(audio_segment.shape[1]):
        channel_data = audio_segment[:, channel]
        analytic_signal = hilbert(channel_data)
        D = np.diff(np.angle(analytic_signal))
        F = np.abs(analytic_signal)
        F_ext = np.random.randn(len(channel_data))
        processed_channel = aphex_equation(D, F, F_ext, alpha)
        processed_channels.append(processed_channel)
    
    return np.column_stack(processed_channels)

def process_audio_file(file_path, alpha):
    audio_data, sample_rate = sf.read(file_path)
    segment_samples = sample_rate * 10  # Process in 10-second segments
    
    processed_segments = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for start in range(0, len(audio_data), segment_samples):
            end = start + segment_samples
            audio_segment = audio_data[start:end]
            futures.append(executor.submit(process_audio_segment, audio_segment, sample_rate, alpha))
        
        for future in futures:
            processed_segments.append(future.result())
    
    processed_audio = np.vstack(processed_segments)
    
    return processed_audio, sample_rate

def main():
    parser = argparse.ArgumentParser(description='Process an audio file using the Aphex equation.')
    parser.add_argument('file_path', type=str, help='Path to the audio file')
    parser.add_argument('--alpha', type=float, default=0.1, help='Decay/release parameter (controls how fast the sound fades)')
    
    args = parser.parse_args()
    
    original_file_name = os.path.splitext(os.path.basename(args.file_path))[0]
    
    try:
        for i in range(21):
            alpha = args.alpha * (i - 10) / 10  # Range from -1.0 to 1.0
            print(f'Processing with alpha: {alpha:.2f}')  # Status indicator
            processed_data, sample_rate = process_audio_file(args.file_path, alpha)
            
            alpha_str = f"{abs(alpha):.2f}".replace('.', '_')
            if alpha < 0:
                alpha_str = f"neg_{alpha_str}"
            else:
                alpha_str = f"pos_{alpha_str}"
            
            output_file_path = f'{original_file_name}_Processed_{alpha_str}.wav'
            sf.write(output_file_path, processed_data, sample_rate)
            print(f'Processed audio saved to {output_file_path}')
    except ValueError as e:
        print(f"ValueError: {e}")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

#
# The aphec.py script is used to visualize the Aphex equation with given parameters. It generates random input signals, applies the Aphex equation, and plots the intermediate steps and the result. The script uses the aphex_equation function from the aphec module to process the signals.
# The apecspec.py script converts an image to an audio signal using frequency modulation. It loads an image, converts it to grayscale, sets up a time array, and processes the image data to generate an audio signal. The script uses the process_chunk and image_to_audio functions to convert the image data to audio.
# The aphspecprovider.py script processes audio data using the Aphex equation. It reads an audio file, processes it in 10-second segments, and applies the Aphex equation to each segment. The script uses the process_audio_segment and process_audio_file functions to process the audio data.
# The main function in the aphec.py script parses command-line arguments, processes an audio file with varying alpha values, and saves the processed audio files with different alpha values. The script handles exceptions and prints status messages during processing.
# The aphec.py script demonstrates the use of the Aphex equation for audio processing, while the apecspec.py script converts images to audio signals. The aphspecprovider.py script processes audio data using the Aphex equation. The main function in the aphec.py script processes audio files with varying alpha values and saves the processed audio files.    
# The main function in the aphec.py script parses command-line arguments, processes an audio file with varying alpha values, and saves the processed audio files with different alpha values. The script handles exceptions and prints status messages during processing.   
# The main function in the aphec.py script parses command-line arguments, processes an audio file with varying alpha values, and saves the processed audio files with different alpha values. The script handles exceptions and prints status messages during processing.
# The main function in the aphec.py script parses command-line arguments, processes an audio file with varying alpha values, and saves the processed audio files with different alpha values. The script handles exceptions and prints status messages during processing. 