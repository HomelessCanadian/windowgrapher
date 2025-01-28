from aphec import aphex_equation
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Set up argument parser
parser = argparse.ArgumentParser(
    description='Visualize the Aphex equation with given parameters.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Decay parameter (alpha)')
parser.add_argument('-f', '--fext_scale', type=float, default=0.1, help='Scale for Far End Cross Talk (F_ext)')
parser.add_argument('-d', '--delay_scale', type=float, default=1.0, help='Scale for Delay (D)')
parser.add_argument('-n', '--num_sets', type=int, default=1, help='Number of random sets to run')

# Parse arguments
try:
    args = parser.parse_args()
except argparse.ArgumentError as e:
    print(f"Argument error: {e}")
    parser.print_help()
    sys.exit(1)

# Run multiple random sets
for set_num in range(args.num_sets):
    # Example input signals
    D = np.random.randn(1, 100) * args.delay_scale  # Scaled random delay values
    F = np.random.randn(5, 100)  # Random frequency values
    F_ext = np.random.randn(100) * args.fext_scale  # Scaled low interference from FEXT

    # Apply the Aphex equation
    result = aphex_equation(D, F, F_ext, args.alpha)

    # Plot the intermediate steps and the result
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(D[0])
    plt.title(f'Delay (D) - Set {set_num + 1}')

    plt.subplot(3, 1, 2)
    plt.plot(F_ext)
    plt.title(f'Far End Cross Talk (F_ext) - Set {set_num + 1}')

    plt.subplot(3, 1, 3)
    plt.plot(result)
    plt.title(f'Aphex Equation Result - Set {set_num + 1}')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

# Example usage:
# python aphec.py -a 0.1 -f 0.1 -d 1.0 -n 3
# This will run 3 random sets of the Aphex equation with the specified parameters. Each set will show the intermediate steps and the result.        
# Compare this snippet from aphspecprovider.py:
# import numpy as np
# from scipy.signal import hilbert
# import soundfile as sf
# from concurrent.futures import ThreadPoolExecutor
#
# def aphex_equation(D, F, F_ext, alpha):
#     return D + F + F_ext * alpha
#
# def process_audio_segment(audio_segment, sample_rate, alpha):
#     processed_channels = []
#     for channel in range(audio_segment.shape[1]):
#         channel_data = audio_segment[:, channel]
#         analytic_signal = hilbert(channel_data)
#         D = np.diff(np.angle(analytic_signal))
#         F = np.abs(analytic_signal)
#         F_ext = np.random.randn(len(channel_data))
#         processed_channel = aphex_equation(D, F, F_ext, alpha)
#         processed_channels.append(processed_channel)
#
#     return np.column_stack(processed_channels)
#
# def process_audio_file(file_path, alpha):
#     audio_data, sample_rate = sf.read(file_path)
#     segment_samples = sample_rate * 10  # Process in 10-second segments
#
#     processed_segments = []
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for start in range(0, len(audio_data), segment_samples):
#             end = start + segment_samples
#             audio_segment = audio_data[start:end]
#             futures.append(executor.submit(process_audio_segment, audio_segment, sample_rate, alpha))
#         for future in futures:
#             processed_segments.append(future.result())
#     processed_audio = np.vstack(processed_segments)
#     return processed_audio, sample_rate
# Compare this snippet from aphspecprovider.py:
# import numpy as np
# from scipy.signal import hilbert
# import soundfile as sf
# from concurrent.futures import ThreadPoolExecutor
#
# def aphex_equation(D, F, F_ext, alpha):
#     return D + F + F_ext * alpha
#
# def process_audio_segment(audio_segment, sample_rate, alpha):
#     processed_channels = []
#     for channel in range(audio_segment.shape[1]):
#         channel_data = audio_segment[:, channel]
#         analytic_signal = hilbert(channel_data)
#         D = np.diff(np.angle(analytic_signal))
#         F = np.abs(analytic_signal)
#         F_ext = np.random.randn(len(channel_data))
#         processed_channel = aphex_equation(D, F, F_ext, alpha)
#         processed_channels.append(processed_channel)
#
#     return np.column_stack(processed_channels)
#
# def process_audio_file(file_path, alpha):
#     audio_data, sample_rate = sf.read(file_path)
#     segment_samples = sample_rate * 10  # Process in 10-second segments
#
#     processed_segments = []
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for start in range(0, len(audio_data), segment_samples):
#             end = start + segment_samples
#             audio_segment = audio_data[start:end]
#             futures.append(executor.submit(process_audio_segment, audio_segment, sample_rate, alpha))
#         for future in futures:
#             processed_segments.append(future.result())
#     processed_audio = np.vstack(processed_segments)
#     return processed_audio, sample_rate
# Compare this snippet from aphspecprovider.py:
# import numpy as np