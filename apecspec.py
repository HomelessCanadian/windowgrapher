from aphspecprovider import aphex_equation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

# Set up argument parser
parser = argparse.ArgumentParser(
    description='Visualize the Aphex equation with given parameters.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Decay parameter (alpha)')
parser.add_argument('-f', '--fext_scale', type=float, default=0.1, help='Scale for Far End Cross Talk (F_ext)')
parser.add_argument('-d', '--delay_scale', type=float, default=1.0, help='Scale for Delay (D)')
parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to the image file')
parser.add_argument('--direct_spectrogram', action='store_true', help='Generate spectrogram without applying the Aphex equation')

# Parse arguments
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Argument error: {e}")
    parser.print_help()
    sys.exit(1)

print("Loading image and converting to grayscale...")
# Load the image and convert to grayscale
image = Image.open(args.image_path).convert('L')
image_data = np.array(image)

print("Setting up time array...")
# Set up the time array
sample_rate = 44100
duration = 5
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

# Convert image to audio signal using frequency modulation with chunk processing
def process_chunk(chunk, t):
    # Reshape frequencies to match time array
    frequencies = 200 + (chunk / 255.0) * 8000  # Map pixel values to frequency range
    frequencies = frequencies.reshape(-1, 1)  # Make column vector
    t = t.reshape(1, -1)  # Make row vector
    
    print(f"Reshaped frequencies: {frequencies.shape}, Reshaped time: {t.shape}")
    return np.sin(2 * np.pi * frequencies * t).sum(axis=0).astype(np.float32)

def image_to_audio(image_data, t, chunk_size=100):
    height, width = image_data.shape
    audio_signal = np.zeros_like(t, dtype=np.float32)
    
    print("Converting image to audio signal...")
    with ThreadPoolExecutor(max_workers=8) as executor:  # Limit the number of threads
        futures = []
        for start in range(0, height, chunk_size):
            end = min(start + chunk_size, height)
            chunk = image_data[start:end, :]
            print(f"Processing rows {start + 1} to {end} of {height}")
            futures.append(executor.submit(process_chunk, chunk, t))
        
        for i, future in enumerate(futures):
            try:
                audio_signal += future.result()
            except Exception as e:
                print(f"Error processing chunk {i + 1}: {e}")
    
    audio_signal = audio_signal / np.max(np.abs(audio_signal))  # Normalize to range [-1, 1]
    return audio_signal

audio_signal = image_to_audio(image_data, t)

# Determine the output filename
original_filename = os.path.splitext(os.path.basename(args.image_path))[0]
output_file = f'{original_filename}_processed_audio.wav'

if args.direct_spectrogram:
    # Save the original audio signal to a WAV file
    write(output_file, 44100, audio_signal.astype(np.float32))
    print(f'Original audio saved to {output_file}')
    
    # Plot the spectrogram of the original audio signal
    frequencies, times, Sxx = spectrogram(audio_signal, fs=44100)
else:
    # Apply the Aphex equation
    D = np.random.randn(1, len(audio_signal)) * args.delay_scale  # Scaled random delay values
    F = np.abs(np.fft.fft(audio_signal))  # Frequency values from FFT
    F_ext = np.random.randn(len(audio_signal)) * args.fext_scale  # Scaled low interference from FEXT
    result = aphex_equation(D, F.reshape(1, -1), F_ext, args.alpha)

    # Save the processed audio signal to a WAV file
    write(output_file, 44100, result.astype(np.float32))
    print(f'Processed audio saved to {output_file}')
    
    # Plot the spectrogram of the processed audio signal
    frequencies, times, Sxx = spectrogram(result, fs=44100)

# Plot the spectrogram
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Audio Signal')
plt.colorbar(label='Intensity [dB]')
plt.show()

