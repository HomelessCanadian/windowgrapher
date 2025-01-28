import numpy as np
from scipy.signal import hilbert
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

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
