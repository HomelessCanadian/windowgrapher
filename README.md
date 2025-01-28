# Windowgrapher

Windowgrapher is a Python-based tool for visualizing and processing audio signals using the Aphex equation. This project includes scripts for converting images to audio signals, processing audio files, and visualizing the results.

## Features

- **Image to Audio Conversion**: Convert an image to an audio signal using frequency modulation.
- **Audio Processing**: Apply the Aphex equation to process audio signals.
- **Visualization**: Visualize the intermediate steps and results of the Aphex equation using matplotlib.

## Installation

To install the necessary dependencies, you can use `pip`:

```sh
pip install numpy matplotlib scipy soundfile pillow
```
## Usage

### Converting Image to Audio

The `apecspec.py` script converts an image to an audio signal:

```
python apecspec.py -a 0.1 -f 0.1 -d 1.0 -i path/to/image.png --direct_spectrogram
```

### Processing Audio Files

The `aphec.py` script processes an audio file using the Aphex equation:

```
python aphec.py path/to/audiofile.wav --alpha 0.1
```

### Running Tests

The `Apectest.py` script runs tests with configurable parameters:

```
python Apectest.py 0.1 0.001
```

### Visualizing the Aphex Equation

The `aphegraph.py` script visualizes the Aphex equation with the given parameters:

```
python aphegraph.py -a 0.1 -f 0.1 -d 1.0 -n 3
```

## Example

Here's an example of converting an image to an audio signal:

```
python apecspec.py -a 0.1 -f 0.1 -d 1.0 -i example_image.png --direct_spectrogram
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
