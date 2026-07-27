"""
QR Code Generator + Reader — Utility Functions
Generate QR codes from text/URLs and decode them back with OpenCV. Fully
self-contained (no dataset): we test the encode->decode round-trip and how small
the code can get before it stops decoding.
"""
import numpy as np
import qrcode
import cv2

_DET = cv2.QRCodeDetector()

def generate(message, box_size=10, border=4):
    """Return a QR code for `message` as a uint8 grayscale numpy image."""
    qr = qrcode.QRCode(box_size=box_size, border=border,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(message); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    return np.array(img, dtype=np.uint8)

def read(image):
    """Decode a QR image back to its string ('' if not decodable)."""
    data, _, _ = _DET.detectAndDecode(image)
    return data

def roundtrip(message, box_size=10):
    """Generate then decode; return (decoded_text, success_bool)."""
    dec = read(generate(message, box_size=box_size))
    return dec, dec == message

def robustness_sweep(messages, box_sizes=(10, 6, 4, 3, 2)):
    """Decode-success count for each QR module size (pixels per module)."""
    return {bs: sum(roundtrip(m, bs)[1] for m in messages) for bs in box_sizes}
