"""Check the delivered MP4's encoding, timing and representative decoded frames."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("video", type=Path)
args = parser.parse_args()
video = args.video
record = json.loads((video.parent / "video_capture_record.json").read_text())
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
reader = imageio.get_reader(str(video))
metadata = reader.get_meta_data()
frames = reader.count_frames()
assert frames == record["frames"] == 720
assert abs(metadata["fps"] - 30) < 1e-6
assert metadata["size"] == (1920, 1080)
assert abs(metadata["duration"] - 24) < .05
thumbnails = []
for i in [0, 105, 315, 495, 645, 719]:
    frame = reader.get_data(i)
    assert np.std(frame[100:-50]) > 5, f"Blank or flat frame {i}"
    thumb = Image.fromarray(frame)
    thumb.thumbnail((640, 360))
    thumbnails.append(thumb)
reader.close()
sheet = Image.new("RGB", (1920, 720))
for i, thumb in enumerate(thumbnails):
    sheet.paste(thumb, ((i % 3) * 640, (i // 3) * 360))
sheet.save(video.parent / "video_contact_sheet.jpg", quality=93)
result = {"decode_passed": True, "frames": frames, "fps": metadata["fps"],
          "duration_s": metadata["duration"], "size": metadata["size"],
          "codec": metadata.get("codec"), "representative_frames_nonblank": True,
          "peg_hold_valid_throughout": record["peg_hold_valid_throughout"]}
(video.parent / "video_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
