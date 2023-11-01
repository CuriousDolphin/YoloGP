from pytube import YouTube
import gradio as gr
from pathlib import Path
import os
from supervision import (
    ImageSink,
    get_video_frames_generator,
    list_files_with_extensions,
)
from tqdm import tqdm
from helpers import zoom_center
import shutil

data_path = Path(__file__).parent.parent / "data"
print("DATA PATH: ", data_path)


def download_youtube_url(url, out_dir) -> str:
    yt = YouTube(url=url)
    files = yt.streams.filter(file_extension="mp4", only_video=True)
    itag = files[0].itag
    video = yt.streams.get_by_itag(int(itag))
    path = video.download(output_path=out_dir)
    return path


def extract_frames(
    url,
    video_path,
    stride,
    start,
    end,
    resize_w,
    zoom,
    progress=gr.Progress(track_tqdm=True),
):
    if video_path is not None:
        v_path = Path(video_path.name)
    elif len(url) > 0:
        progress(0.1, "Downloading..")
        d_path = download_youtube_url(url, data_path)
        v_path = Path(d_path)
    print("video path:", v_path)
    video_name = str(v_path.stem).replace(" ", "")
    target_dir = Path(f"{data_path}/{video_name}_frames")
    cont = 0
    with ImageSink(
        target_dir_path=target_dir,
        image_name_pattern="image_{:05d}.jpg",
        overwrite=True,
    ) as sink:
        for image in tqdm(
            get_video_frames_generator(
                source_path=str(v_path), stride=stride, start=start
            )
        ):
            if zoom > 1:
                image = zoom_center(img=image.copy(), zoom_factor=zoom)
            sink.save_image(image=image.copy())
            cont += 1
    progress(0.8, "Zipping..")
    print("Target_dir", target_dir)
    frames = list_files_with_extensions(directory=target_dir, extensions=["jpg", "png"])
    print(len(frames))
    archive_ = shutil.make_archive(
        target_dir,
        "zip",
        target_dir,
    )

    print(archive_)

    v_path.unlink()
    return frames[0:10], [archive_]


inputs = [
    gr.Textbox(label="Youtube_url"),
    gr.File(label="mp4 or mov", file_types=["video"]),
    gr.Slider(label="Stride", value=60, maximum=1200),
    gr.Number(label="Start Frame", value=0),
    gr.Number(label="End Frame", value=-1),
    gr.Number(label="Resize Width (px)", value=-1),
    gr.Slider(label="Image Zoom", minimum=1.0, maximum=2.99, value=1.4),
]
outputs = [gr.Gallery(label="preview"), gr.File()]
interface = gr.Interface(
    fn=extract_frames,
    inputs=inputs,
    outputs=outputs,
    examples=[["https://www.youtube.com/watch?v=XDhjS_fzhsQ"]],
    allow_flagging=False,
)


if __name__ == "__main__":
    interface.queue(max_size=10).launch(server_name="0.0.0.0")
