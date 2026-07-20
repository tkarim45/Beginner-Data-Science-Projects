# QR Code Generator + Reader

A beginner-level **applied** project: a small, fully self-contained tool that encodes text/URLs into QR
codes and decodes them back with OpenCV, then measures how robust the round-trip is.

## Problem Statement
Generate a QR code from any string, decode it reliably, and answer the practical questions: does the
round-trip survive different payload types, how small can the code be before it fails, and how much noise
can it tolerate? No dataset needed, the tool generates its own inputs.

## Dataset
- **None**, fully self-contained. Test payloads (URL, text, WiFi config, vCard, numbers, unicode) are
  defined in the notebooks and generated on the fly.

## Project Structure
```
QR Code Generator and Reader/
├── 01_generate_and_read.ipynb  # Generate a QR, decode it, round-trip on varied payloads
├── 02_robustness.ipynb         # Decode success vs module size, and under salt-and-pepper noise
├── utils.py · requirements.txt · README.md
└── data/  (none)
```

## Method
`qrcode` (with Reed, Solomon error correction level M) to generate, OpenCV `QRCodeDetector` to decode.
`roundtrip()` encodes then decodes; `robustness_sweep()` shrinks the module size until decoding breaks.

## Key Findings
All figures produced by executing the notebooks, not assumed.
- **8/8 payloads round-trip perfectly** at normal size. URLs, plain text, structured WiFi/vCard payloads,
  numbers, and unicode all encode and decode correctly.
- **QR codes are remarkably size-robust**, all payloads still decode down to **2 pixels per module**;
  only at 1px/module does decoding break.
- **Error correction tolerates moderate noise**, decoding survives a few percent of corrupted pixels
  before degrading, thanks to the Reed, Solomon error correction built into the QR standard.
- **Practical guidance**: keep ≥3 to 4 px/module when printing/displaying for a safety margin; higher
  error-correction levels trade data capacity for resilience.

## Tech Stack
- numpy, matplotlib, qrcode, opencv-python

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook 01_generate_and_read.ipynb
```
