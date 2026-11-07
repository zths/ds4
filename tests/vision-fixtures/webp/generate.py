"""Regenerate synthetic WebP/PNG pairs; print fingerprints for test_image_decode.c."""
from pathlib import Path
import hashlib, struct
from PIL import Image, features

root = Path(__file__).resolve().parent
width, height = 16, 12
image = Image.new('RGBA', (width, height))
image.putdata([((x*17+y*7) % 256, (x*11+y*19) % 256, (x*3+y*23) % 256,
                [0, 64, 128, 255][(x+y) % 4])
               for y in range(height) for x in range(width)])
print('Reference libwebp:', features.version_module('webp'))
for name, mode, options in [
    ('lossless', 'RGB', dict(lossless=True, exact=True)),
    ('alpha', 'RGBA', dict(lossless=True, exact=True)),
    ('lossy', 'RGB', dict(lossless=False, quality=75)),
]:
    image.convert(mode).save(root / (name + '.webp'), format='WEBP', **options)
    with Image.open(root / (name + '.webp')) as decoded:
        rgb = decoded.convert('RGB')
        rgb.save(root / (name + '.png'))
        print(name, hashlib.sha256(struct.pack('<II', width, height) + rgb.tobytes()).hexdigest())
image.save(root / 'animated.webp', save_all=True,
           append_images=[Image.new('RGBA', (width, height), 'green')],
           format='WEBP', lossless=True, duration=100, loop=0)
