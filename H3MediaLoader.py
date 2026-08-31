"""H3MediaLoader：在一个节点中聚合可选的图像、音频与文本描述。

图像加载行为对齐 LoadImage（RGB 张量，动图按帧拼接，带 input 下拉）。
音频加载行为对齐 H3AudioUpload（波形裁剪在执行时生效，不改磁盘文件）。
文本为多行描述；三项均可为空，空项在 media 对象中为 None。
同时输出 media 以及独立的 image / audio / prompt，便于接入其它节点。
"""

import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths


# 图像下拉中的空选项，表示不加载图像。
IMAGE_NONE = "(none)"

# 无 content-type 过滤时，按扩展名识别图像文件。
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".jpe", ".webp", ".gif",
    ".bmp", ".tif", ".tiff", ".apng",
}


def _is_empty_name(name):
    """判断文件名是否视为未选择（空、空白或 (none)）。"""
    if name is None:
        return True
    text = str(name).strip()
    return text == "" or text in (IMAGE_NONE, "none", "None")


def _list_input_images():
    """列出 input 目录中的图像文件，供 LoadImage 风格下拉使用。"""
    names = [IMAGE_NONE]
    try:
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        if hasattr(folder_paths, "filter_files_content_types"):
            files = folder_paths.filter_files_content_types(files, ["image"])
        else:
            files = [
                f for f in files
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]
        names.extend(sorted(files))
    except Exception:
        pass
    return names


def _file_signature(file_path):
    """生成文件路径与修改时间、大小的签名，供 IS_CHANGED 使用。"""
    if not file_path:
        return ""
    try:
        stat = os.stat(file_path)
        return "%s:%s:%s" % (file_path, stat.st_mtime, stat.st_size)
    except OSError:
        return str(file_path)


def _resolve_input_path(filename):
    """在 ComfyUI input 目录中解析上传文件的实际路径。"""
    if _is_empty_name(filename):
        return None
    filename = str(filename).strip()

    try:
        annotated = folder_paths.get_annotated_filepath(filename)
        if annotated and os.path.exists(annotated):
            return annotated
    except Exception:
        pass

    input_dir = folder_paths.get_input_directory()
    candidates = [
        os.path.join(input_dir, filename),
        filename if os.path.isabs(filename) else None,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _normalize_text(text):
    """空字符串视为未填写，返回 None；其它文本原样保留。"""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    return text if text != "" else None


def _normalize_role_name(role_name):
    """归一化角色名称，空白名称返回 None，非空名称去除首尾空白。"""
    if role_name is None:
        return None
    if not isinstance(role_name, str):
        role_name = str(role_name)
    role_name = role_name.strip()
    return role_name if role_name else None


def _load_image_tensor(file_path):
    """按 LoadImage 方式读取图像，返回 [B, H, W, C] float32 张量。"""
    img = Image.open(file_path)
    output_images = []
    width = None
    height = None
    excluded_formats = ["MPO"]

    for frame in ImageSequence.Iterator(img):
        frame = ImageOps.exif_transpose(frame)
        if frame.mode == "I":
            frame = frame.point(lambda i: i * (1 / 255))
        image = frame.convert("RGB")

        if width is None:
            width, height = image.size
        if image.size[0] != width or image.size[1] != height:
            continue
        if frame.format in excluded_formats:
            continue

        array = np.array(image).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(array)[None, ...])

        if img.format not in ("GIF", "WEBP", "APNG"):
            break

    if hasattr(img, "close"):
        img.close()

    if not output_images:
        raise RuntimeError("H3MediaLoader: failed to decode image: %s" % file_path)
    return torch.cat(output_images, dim=0)


def _load_wav_with_wave(file_path):
    """用标准库 wave 读取 PCM WAV，作为 torchaudio/soundfile 的回退。"""
    import wave
    with wave.open(str(file_path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        sampwidth = handle.getsampwidth()
        nframes = handle.getnframes()
        raw = handle.readframes(nframes)
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("unsupported WAV sample width: %s" % sampwidth)
    if channels > 1:
        data = data.reshape(-1, channels).T
    else:
        data = data.reshape(1, -1)
    return torch.from_numpy(np.ascontiguousarray(data)).float(), int(sr)


def _load_audio_dict(file_path, trim_start, trim_end):
    """读取音频并按秒级裁剪范围返回 ComfyUI AUDIO 字典。"""
    wav = None
    sr = None
    last_error = None

    try:
        import torchaudio
        wav, sr = torchaudio.load(str(file_path))
    except Exception as exc:
        last_error = exc
        try:
            import soundfile as sf
            data, sr = sf.read(str(file_path), always_2d=True)
            wav = torch.from_numpy(data.T).float()
        except Exception as inner:
            last_error = inner
            try:
                wav, sr = _load_wav_with_wave(file_path)
            except Exception as wave_exc:
                raise RuntimeError(
                    "H3MediaLoader: failed to decode audio file %s: %s"
                    % (file_path, last_error)
                ) from wave_exc

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    wav = wav.float()

    total_samples = wav.shape[-1]
    if total_samples <= 0:
        raise RuntimeError("H3MediaLoader: audio file is empty.")

    if trim_end is None or trim_end < 0:
        end_sample = total_samples
    else:
        end_sample = int(round(float(trim_end) * sr))
    start_sample = int(round(float(trim_start or 0.0) * sr))
    start_sample = max(0, min(start_sample, total_samples - 1))
    end_sample = max(start_sample + 1, min(end_sample, total_samples))
    trimmed = wav[..., start_sample:end_sample]
    return {"waveform": trimmed, "sample_rate": int(sr)}


class H3MediaLoader:
    """可选图像 / 音频 / prompt 的聚合加载节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明图像下拉、隐藏的音频控件，以及可见的多行 prompt。"""
        return {
            "required": {
                "image_filename": (_list_input_images(), {
                    "image_upload": True,
                    "default": IMAGE_NONE,
                    "tooltip": "从 input 目录选择图像，可为空。",
                }),
                "audio_filename": ("STRING", {"default": ""}),
                "trim_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 99999.0,
                    "step": 0.01,
                }),
                "trim_end": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 99999.0,
                    "step": 0.01,
                }),
                "audio_muted": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "静音时不输出 audio。",
                }),
                "role_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "角色名称，可为空。",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "角色提示说明，可为空。",
                }),
            },
        }

    RETURN_TYPES = ("H3_MEDIA", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("media", "image", "audio", "prompt")
    FUNCTION = "load_media"
    CATEGORY = "H3Nodes"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "在一个节点中可选地加载图像、音频（波形预览/裁剪）和描述 prompt，"
        "同时输出 media 对象以及独立的 image / audio / prompt。"
    )

    def load_media(
        self,
        image_filename="",
        audio_filename="",
        trim_start=0.0,
        trim_end=-1.0,
        audio_muted=False,
        role_name="",
        prompt="",
    ):
        """加载可选的图像、音频、角色名称与 prompt，同时返回 media 与独立输出。"""
        image = None
        audio = None

        image_name = "" if _is_empty_name(image_filename) else str(image_filename).strip()
        if image_name:
            image_path = _resolve_input_path(image_name)
            if image_path is None:
                raise FileNotFoundError(
                    "H3MediaLoader: image file not found: %s" % image_filename
                )
            try:
                image = _load_image_tensor(image_path)
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    "H3MediaLoader: failed to load image %s: %s"
                    % (image_path, exc)
                ) from exc

        audio_name = (audio_filename or "").strip()
        if audio_name and not bool(audio_muted):
            audio_path = _resolve_input_path(audio_name)
            if audio_path is None:
                raise FileNotFoundError(
                    "H3MediaLoader: audio file not found: %s" % audio_filename
                )
            audio = _load_audio_dict(audio_path, trim_start, trim_end)

        normalized_role_name = _normalize_role_name(role_name)
        normalized_prompt = _normalize_text(prompt)
        media = {
            "image": image,
            "audio": audio,
            "role_name": normalized_role_name,
            "prompt": normalized_prompt,
        }
        # STRING 输出用空串代替 None，便于直接接入文本类节点。
        prompt_out = normalized_prompt if normalized_prompt is not None else ""
        return (media, image, audio, prompt_out)

    @classmethod
    def IS_CHANGED(
        cls,
        image_filename="",
        audio_filename="",
        trim_start=0.0,
        trim_end=-1.0,
        audio_muted=False,
        role_name="",
        prompt="",
    ):
        """根据文件签名、裁剪、静音、角色名称与 prompt 内容判断是否需要重新执行。"""
        image_path = _resolve_input_path(image_filename)
        audio_path = _resolve_input_path(audio_filename)
        return "|".join([
            _file_signature(image_path) or str(image_filename or ""),
            _file_signature(audio_path) or str(audio_filename or ""),
            str(trim_start),
            str(trim_end),
            str(bool(audio_muted)),
            role_name or "",
            prompt or "",
        ])


NODE_CLASS_MAPPINGS = {
    "H3MediaLoader": H3MediaLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MediaLoader": "H3 Media Loader",
}

NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
