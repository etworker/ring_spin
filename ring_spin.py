# -*- coding: utf-8 -*-
"""
This script generates a "ring spin" visual effect on an image. It divides the
image into several concentric rings around a specified center point and rotates
each ring by a given angle over a series of steps, creating an animation.

The script can output the result in various formats, including a static image (JPG),
an animated GIF, a video (MP4), and an Apple Live Photo. It also supports
adding a looping audio track to the video outputs.

Dependencies:
- numpy
- Pillow (PIL)
- opencv-python
- FFmpeg (must be installed and in the system's PATH)
- pillow-heif (optional, for HEIC image support)
- makelive (optional, for Live Photo creation)

Usage as a library:
--------------------
from ring_spin_effect import ring_spin

ring_spin(
    image_path="path/to/your/image.jpg",
    center_rel_pos=(0.5, 0.5),
    num_rings=8,
    ring_width=100,
    rotate_angles=[10, 20, 30, 40, 40, 30, 20, 10],
    num_steps=10,
    audio_path="path/to/your/audio.wav",
    duration_in_sec=3.0,
    output_dir="output/directory",
    output_name="my_effect",
    output_formats=["jpg", "gif", "mp4", "live"]
)

Usage from the command line:
-----------------------------
python ring_spin_effect.py data/in/chimney.jpg --num-rings 10 --ring-width 80 --output-formats mp4 gif
"""

import math
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import sys
import shutil
import subprocess

from typing import List, Tuple, Optional

# - Import support for HEIC format
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _create_concatenated_audio(audio_path: str, num_repeats: int, output_dir: str) -> str:
    """
    Creates an audio file by repeating the original audio multiple times.

    Args:
        audio_path: Path to the source audio file.
        num_repeats: The number of times to repeat the audio.
        output_dir: Directory to store temporary and output files.

    Returns:
        The file path of the concatenated audio file.
    """

    temp_audio_list_path = os.path.join(output_dir, "audio_list.txt")
    concatenated_audio_path = os.path.join(output_dir, "concatenated_audio.wav")

    # - Create a file list required by ffmpeg's concat demuxer
    with open(temp_audio_list_path, "w") as f:
        for _ in range(num_repeats):
            f.write(f"file '{os.path.abspath(audio_path)}'\n")

    # - Use ffmpeg's concat demuxer to join the audio files
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        temp_audio_list_path,
        "-c",
        "copy",
        concatenated_audio_path,
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # - Delete the temporary file list
    os.remove(temp_audio_list_path)

    return concatenated_audio_path


def ring_spin(
    image_path: str,
    center_rel_pos: Tuple[float, float] = (0.5, 0.5),
    num_rings: int = 8,
    ring_width: int = 100,
    rotate_angles: List[int] = [10, 20, 30, 40, 40, 30, 20, 10],
    num_steps: int = 10,
    audio_path: Optional[str] = None,
    duration_in_sec: float = 3.0,
    output_dir: str = "data/out",
    output_name: str = "",
    output_formats: List[str] = ["jpg", "gif", "mp4", "live"],
) -> List[np.ndarray]:
    """
    Applies a "ring spin" effect to an image, generating a sequence of frames.

    Args:
        image_path: Path to the input image.
        center_rel_pos: Relative coordinates of the circle's center, defaults to (0.5, 0.5) for the image center.
        num_rings: Number of concentric rings, defaults to 8.
        ring_width: Width of each ring in pixels, defaults to 100.
        rotate_angles: A list of final rotation angles for each ring, defaults to [10, 20, 30, 40, 40, 30, 20, 10].
        num_steps: Number of steps for the rotation animation, defaults to 10.
        audio_path: Path to an audio file to be played with each step.
        duration_in_sec: Total duration of the output animation in seconds, defaults to 3.0.
        output_dir: Directory to save the output files.
        output_name: Base name for the output files. If empty, uses the input image's name.
        output_formats: A list of output formats to generate. Supported formats: ["jpg", "gif", "mp4", "live"].

    Returns:
        A list of numpy arrays, where each array is a frame of the generated animation.
    """

    # - Load the image and convert to a numpy array
    img = Image.open(image_path)
    img_array = np.array(img)

    # - Get image dimensions
    height, width = img_array.shape[:2]

    # - Calculate the absolute center coordinates
    center_x = int(width * center_rel_pos[0])
    center_y = int(height * center_rel_pos[1])

    # - Adjust the rotate_angles list to match the number of rings
    if len(rotate_angles) != num_rings:
        if len(rotate_angles) < num_rings:
            rotate_angles = rotate_angles + [0] * (num_rings - len(rotate_angles))
        else:
            rotate_angles = rotate_angles[:num_rings]

    # - Pre-computation phase for optimization
    # - This calculates ring masks and polar coordinates once before generating frames.
    y, x = np.ogrid[:height, :width]
    dist_from_center_sq = (x - center_x) ** 2 + (y - center_y) ** 2

    precomputed_rings = []
    for i in range(num_rings):
        outer_radius_sq = ((i + 1) * ring_width) ** 2
        inner_radius_sq = (i * ring_width) ** 2

        # - Create a mask for the current ring
        ring_mask = (dist_from_center_sq <= outer_radius_sq) & (dist_from_center_sq > inner_radius_sq)

        if not np.any(ring_mask):
            precomputed_rings.append(None)
            continue

        # - Get coordinates of the pixels within the ring
        ring_coords = np.where(ring_mask)

        # - Convert Cartesian coordinates to polar coordinates (relative to the center)
        rel_x = ring_coords[1] - center_x
        rel_y = ring_coords[0] - center_y
        angles_polar = np.arctan2(rel_y, rel_x)
        radii_polar = np.sqrt(rel_x**2 + rel_y**2)

        precomputed_rings.append({"coords": ring_coords, "angles_polar": angles_polar, "radii_polar": radii_polar})

    # - Frame generation
    frames = []
    # - Note: Loop includes step 0 (original image) and goes up to num_steps.
    for step in range(num_steps + 1):
        # - Create a copy of the original image for modification
        new_img = img_array.copy()

        # - Calculate the rotation angles for the current step
        current_angles = [angle * step / num_steps for angle in rotate_angles]

        # - Process each ring from the outermost to the innermost to prevent overlap issues
        for i in range(num_rings - 1, -1, -1):
            ring_data = precomputed_rings[i]
            if ring_data is None:
                continue

            # - Get the rotation angle for the current ring and step
            rotation_angle_rad = math.radians(current_angles[i])

            # - Apply rotation to the polar angles
            new_angles = ring_data["angles_polar"] - rotation_angle_rad

            # - Convert polar coordinates back to Cartesian coordinates
            new_x = center_x + ring_data["radii_polar"] * np.cos(new_angles)
            new_y = center_y + ring_data["radii_polar"] * np.sin(new_angles)

            # - Round to the nearest integer to get pixel coordinates
            new_x_int = np.round(new_x).astype(int)
            new_y_int = np.round(new_y).astype(int)

            # - Create a mask for coordinates that are within the image boundaries
            valid_coords_mask = (new_x_int >= 0) & (new_x_int < width) & (new_y_int >= 0) & (new_y_int < height)

            # - Filter out coordinates that fall outside the image
            src_coords_y = ring_data["coords"][0][valid_coords_mask]
            src_coords_x = ring_data["coords"][1][valid_coords_mask]
            new_y_valid = new_y_int[valid_coords_mask]
            new_x_valid = new_x_int[valid_coords_mask]

            # - Update the new image with pixels from the rotated source
            new_img[src_coords_y, src_coords_x] = img_array[new_y_valid, new_x_valid]

        # - Draw ring boundaries for visualization
        img_with_borders = Image.fromarray(new_img)
        draw = ImageDraw.Draw(img_with_borders)

        for i in range(num_rings + 1):
            radius = i * ring_width
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                outline="white",
                width=1,
            )

        frames.append(np.array(img_with_borders))

    os.makedirs(output_dir, exist_ok=True)

    if not output_name:
        output_name = os.path.splitext(os.path.basename(image_path))[0]

    # - Prepare the concatenated audio file if needed
    final_audio_path = None
    if audio_path and os.path.exists(audio_path) and ("mp4" in output_formats or "live" in output_formats):
        # - We have num_steps of rotation, so we need to play the sound num_steps times
        final_audio_path = _create_concatenated_audio(audio_path, num_steps, output_dir)

    for fmt in output_formats:
        if fmt == "jpg":
            output_path = os.path.join(output_dir, f"{output_name}.jpg")

            # - If the output path is the same as the input, add a timestamp to avoid overwriting
            if os.path.abspath(output_path) == os.path.abspath(image_path):
                from datetime import datetime

                now = datetime.now()
                time_str = now.strftime("%Y%m%d%H%M%S")
                output_path = os.path.join(output_dir, f"{output_name}_{time_str}.jpg")

            Image.fromarray(frames[-1]).save(output_path)
            print(f"Saved final image to: {output_path}")

        elif fmt == "gif":
            output_path = os.path.join(output_dir, f"{output_name}.gif")
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_in_sec * 1000 / len(frames),
                loop=0,
            )
            print(f"Saved GIF animation to: {output_path}")

        elif fmt == "mp4":
            output_path = os.path.join(output_dir, f"{output_name}.mp4")
            # - Use the full list of frames, including the initial static frame
            video_frames = frames
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = len(video_frames) / duration_in_sec
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()

            if final_audio_path:
                temp_video_path = os.path.join(output_dir, f"{output_name}_temp.mp4")
                os.rename(output_path, temp_video_path)

                # - Use the -shortest option to ensure the output duration matches the shorter stream (video)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_video_path,
                    "-i",
                    final_audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                    output_path,
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(temp_video_path)

            print(f"Saved MP4 video to: {output_path}")

        elif fmt == "live":
            temp_dir = os.path.join(output_dir, f"{output_name}_temp_live")
            os.makedirs(temp_dir, exist_ok=True)

            # - The video part of a Live Photo should not include the static frame 0
            video_frames = frames[1:]
            video_path = os.path.join(temp_dir, f"{output_name}.mov")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # - The video duration should be the total duration minus one frame's time
            video_duration = duration_in_sec * (len(video_frames) / len(frames))
            fps = len(video_frames) / video_duration if video_duration > 0 else 30
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in video_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()

            if final_audio_path:
                temp_video_path = os.path.join(temp_dir, f"{output_name}_temp.mov")
                os.rename(video_path, temp_video_path)

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_video_path,
                    "-i",
                    final_audio_path,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                    video_path,
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(temp_video_path)

            # - Use the makelive library to create the Live Photo
            try:
                from makelive import save_live_photo_pair_as_pvt

                photo_path = os.path.join(temp_dir, f"{output_name}.jpg")
                Image.fromarray(frames[0]).save(photo_path)

                asset_id, pvt_path = save_live_photo_pair_as_pvt(photo_path, video_path)

                final_pvt_path = os.path.join(output_dir, os.path.basename(pvt_path))
                if os.path.exists(final_pvt_path):
                    os.remove(final_pvt_path)
                shutil.move(pvt_path, output_dir)
                print(f"Saved Live Photo to: {final_pvt_path}, asset_id: {asset_id}")

            except ImportError:
                print(
                    "Error: 'makelive' library is not installed. Cannot create Live Photo. Please run 'pip install makelive'"
                )
            finally:
                shutil.rmtree(temp_dir)

    # - Clean up the concatenated audio file
    if final_audio_path and os.path.exists(final_audio_path):
        os.remove(final_audio_path)

    return frames


def main():
    # - If this script is run directly, process a default image.
    image_path = "data/in/chimney.jpg"
    # image_path = "data/in/tree.HEIC"
    ring_spin(
        image_path=image_path,
        audio_path="data/in/click.wav",
        output_formats=["jpg", "gif", "mp4", "live"],
    )


if __name__ == "__main__":
    # - If command-line arguments are provided, parse them
    if len(sys.argv) > 1:
        import argparse

        parser = argparse.ArgumentParser(description="Ring Spin Effect Generator")
        parser.add_argument("image_path", help="Path to the input image.")
        parser.add_argument(
            "--center_rel_pos",
            nargs=2,
            type=float,
            default=[0.5, 0.5],
            help="Relative coordinates of the circle's center (e.g., 0.5 0.5 for the center).",
        )
        parser.add_argument("--num-rings", type=int, default=8, help="Number of concentric rings.")
        parser.add_argument("--ring-width", type=int, default=100, help="Width of each ring in pixels.")
        parser.add_argument(
            "--rotate-angles",
            nargs="+",
            type=int,
            default=[10, 20, 30, 40, 40, 30, 20, 10],
            help="List of final rotation angles for each ring.",
        )
        parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for the animation.")
        parser.add_argument("--audio-path", help="Path to an audio file to be played with each step.")
        parser.add_argument(
            "--duration_in_sec", type=float, default=3.0, help="Total duration of the animation in seconds."
        )
        parser.add_argument("--output-dir", default="data/out", help="Directory to save the output files.")
        parser.add_argument(
            "--output-name", default="", help="Base name for output files (defaults to input file name)."
        )
        parser.add_argument(
            "--output-formats",
            nargs="+",
            default=["jpg", "gif", "mp4", "live"],
            help='List of output formats. Default: ["jpg", "gif", "mp4", "live"]',
        )

        args = parser.parse_args()

        # - Call the main function with parsed arguments
        ring_spin(
            image_path=args.image_path,
            center_rel_pos=tuple(args.center_rel_pos),
            num_rings=args.num_rings,
            ring_width=args.ring_width,
            rotate_angles=args.rotate_angles,
            num_steps=args.num_steps,
            audio_path=args.audio_path,
            duration_in_sec=args.duration_in_sec,
            output_dir=args.output_dir,
            output_name=args.output_name,
            output_formats=args.output_formats,
        )
    else:
        main()
