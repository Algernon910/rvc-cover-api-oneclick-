import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO

from typing import Union

def wav2(i, o, format):
    # PyAV 10 之后 av.open 只认 "r"/"w"，不再接受 "rb"/"wb"
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    if not inp.streams.audio:
        raise ValueError("Input contains no audio stream")
    input_stream = inp.streams.audio[0]
    source_rate = input_stream.codec_context.sample_rate
    ostream = (
        out.add_stream(format, rate=source_rate)
        if source_rate
        else out.add_stream(format)
    )
    # PyAV 15 里 AudioCodecContext.channels 只读，声道数通过 layout 设置
    source_channels = input_stream.codec_context.channels
    if source_channels == 1:
        ostream.layout = "mono"
    elif source_channels == 2:
        ostream.layout = "stereo"

    for frame in inp.decode(input_stream):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    # 采样率与声道数都要在 add_stream / layout 上设置（PyAV 15 下 channels 只读）
    ostream = out.add_stream(format, rate=sr)
    ostream.layout = "mono"

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file: Union[str, bytes], sr: int):
    # open audio file
    if type(file) == bytes:
        opened_file = BytesIO(file)
    else:
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        opened_file = open(file, "rb")

    # convert audio file to 16k wav
    try:
        with BytesIO() as out:
            audio2(opened_file, out, "f32le", sr)
            ret = np.frombuffer(out.getvalue(), np.float32).flatten()
    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        ret = librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except:
        raise RuntimeError(traceback.format_exc())

    # close audio file to be safe
    opened_file.close()

    return ret
