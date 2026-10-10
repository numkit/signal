# numkit/signal

[![License: 0BSD](https://img.shields.io/badge/License-0BSD-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)

The **Signal Processing Toolbox** for **[NumKit](https://github.com/numkit/numkit)**.

High-performance digital signal processing algorithms with Highway SIMD vectorization and MATLAB R2025b parity.

---

## Features

- **Spectral & Transforms**: FFT, IFFT, 2D/N-D FFT, DCT, DST, FWHT, Goertzel, Hilbert, CZT.
- **Digital Filtering**: IIR/FIR filter implementation (Direct Form II Transposed), forward-backward zero-phase filtering (`filtfilt`), 2D filtering.
- **Filter Design**: Butterworth (`butter`), Chebyshev I/II (`cheby1`, `cheby2`), Elliptic (`ellip`), Bessel (`besself`), windowed FIR (`fir1`, `fir2`), least-squares FIR (`firls`).
- **Filter Analysis**: Frequency response (`freqz`, `phasez`, `grpdelay`), unwrap.
- **Windows**: Kaiser, Hamming, Hann, Blackman, Chebyshev, Tukey, Bartlett, Triangular.
- **Waveform Generation**: Chirp, square, sawtooth, sinc, diric, Gaussian pulse.
- **Measurements & Peaks**: `findpeaks`, SNR, THD, SINAD, peak-to-peak, RMS.
- **Multirate**: Resample (polyphase filter), upsample, downsample, decimate.

---

## Building as a Dynamic Module

```bash
cmake -B build -S .
cmake --build build --config Release
```

Produces `nk_signal.dll` (Windows) or `nk_signal.so` (Linux/macOS).

---

## Usage in NumKit

Install via the NumKit package manager:

```bash
nk pkg install signal
```

Or load dynamically in a script:

```matlab
% Loaded automatically on first call to any signal function:
[b, a] = butter(4, 0.2);
[h, w] = freqz(b, a, 512);
plot(w, 20*log10(abs(h)));
```

---

## License

This software is released under the **[BSD Zero Clause License (0BSD)](LICENSE)**.
