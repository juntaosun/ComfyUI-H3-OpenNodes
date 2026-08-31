import logging

import torch

_LOG = logging.getLogger("H3VideoFrom")

# ==========================================================================
# 主要用于在下游支持接入 MiniMaxH3AddGuide 方便使用首尾帧控制视频生成
# ==========================================================================

# H3 关键帧栅格：实际帧数 = 17 * k + 5，例如 5、22、39...
_H3_FRAME_BASE = 5
_H3_FRAME_STEP = 17


def _snap_h3_frames(frame_count):
    """将请求帧数就近对齐到合法的 17k+5，最小为 5。"""
    frame_count = int(frame_count)
    if frame_count <= _H3_FRAME_BASE:
        return _H3_FRAME_BASE
    k = int(round((frame_count - _H3_FRAME_BASE) / float(_H3_FRAME_STEP)))
    return _H3_FRAME_STEP * max(0, k) + _H3_FRAME_BASE


def _floor_h3_frames(frame_count):
    """将帧数向下对齐到不超过自身的最大 17k+5。

    不足 5 帧时仍返回 5，由调用方决定是否补帧。
    """
    frame_count = int(frame_count)
    if frame_count < _H3_FRAME_BASE:
        return _H3_FRAME_BASE
    k = (frame_count - _H3_FRAME_BASE) // _H3_FRAME_STEP
    return _H3_FRAME_STEP * k + _H3_FRAME_BASE


def _resolve_keep(requested, available):
    """根据可用长度决定实际取出的 17k+5 帧数。

    requested 已是合法栅格；available 不足时下压到更大不超过可用长度的栅格。
    available <= 0 时返回 0。
    """
    requested = _snap_h3_frames(requested)
    available = int(available)
    if available <= 0:
        return 0
    if available >= requested:
        return requested
    return _floor_h3_frames(available)


def _pad_images(images, keep):
    """将图像序列补到 keep 帧，不足部分重复最后一帧；空序列原样返回。"""
    total = int(images.shape[0])
    if total <= 0 or keep <= total:
        return images
    last = images[-1:].expand(keep - total, *images.shape[1:]).clone()
    return torch.cat([images, last], dim=0)


def _slice_audio(waveform, sample_rate, fps, keep, from_end):
    """按 keep 帧对应时长从波形头或尾取出音频，并精确对齐到 frames/fps。

    采样点数用 round(keep / fps * sr) 计算，与 H3VideoAudioCut 一致。
    波形偏长则截断，偏短则尾部补零。
    """
    sample_rate = int(sample_rate)
    keep = max(0, int(keep))
    want = int(round(keep / float(fps) * sample_rate)) if keep else 0
    length = int(waveform.shape[-1])
    if want <= 0:
        return waveform[..., :0]
    if from_end:
        start = max(0, length - want)
        piece = waveform[..., start:]
    else:
        piece = waveform[..., :min(want, length)]
    have = int(piece.shape[-1])
    if have > want:
        return piece[..., :want]
    if have < want:
        return torch.nn.functional.pad(piece, (0, want - have))
    return piece


def _image_count(images):
    """返回图像实际帧数；空输入或空张量记为 0。"""
    if images is None:
        return 0
    return max(0, int(images.shape[0]))


def _resize_ref_image(ref_image, target_height, target_width):
    """将参考图等比缩放并居中放入目标尺寸的纯白画布。"""
    if ref_image is None or int(ref_image.shape[0]) <= 0:
        return None
    image = ref_image[:1]
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    if source_height <= 0 or source_width <= 0:
        return None
    if (source_height, source_width) == (target_height, target_width):
        return image[0]

    scale = min(target_width / float(source_width),
                target_height / float(source_height))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    resized_width = max(1, min(target_width, round(source_width * scale)))
    samples = image.permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        samples,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    # 以纯白画布补齐等比缩放后多出的边缘，保持参考图居中且不裁剪。
    canvas = torch.ones(
        (target_height, target_width, image.shape[3]),
        dtype=image.dtype,
        device=image.device,
    )
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _add_sampling_noise(images, noise_strength=0.5, exclude_index=None):
    """按指定强度混合标准高斯噪声，支持标量或逐帧强度序列。"""
    if images is None or int(images.shape[0]) <= 0:
        return images

    # 将噪声强度限制在有效范围，0 表示不加噪，1 表示完全使用噪声。
    strength = torch.as_tensor(
        noise_strength, dtype=images.dtype, device=images.device)
    if strength.numel() == 1:
        strength = strength.reshape(1, 1, 1, 1)
    else:
        if strength.numel() != int(images.shape[0]):
            raise ValueError("逐帧噪声强度数量必须与图像帧数一致")
        strength = strength.reshape(-1, 1, 1, 1)
    strength = strength.clamp(0.0, 1.0)
    # 保留一份清晰图像用于参考帧排除，避免噪声操作修改输入张量。
    source_images = images.clone()
    noise = torch.randn_like(source_images)
    noisy_images = source_images * (1.0 - strength) + noise * strength
    noisy_images = noisy_images.clamp(0.0, 1.0)
    if exclude_index is not None:
        noisy_images[exclude_index] = source_images[exclude_index]
    return noisy_images


class H3VideoAudioFrom:
    """从完整视频/音频中按 H3 关键帧栅格取出开头或尾部片段。

    H3 视频模型按 17k+5 帧成组，因此 from_frames 与返回图像帧数都会
    对齐到 5、22、39... 。beginning 取片头，end 取片尾；音频按时长
    同步截取，并用 fps 将采样数精确对齐到取出的帧数。
    keep_audio=False 时跳过音频，直接返回 None。
    count 为实际图像帧数；frame_idx 在 beginning 为正、end 为负。
    """

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：可选图像/音频、栅格帧数、取段方向、帧率与是否保留音频。"""
        return {
            "required": {
                "from_frames": ("INT", {
                    "default": 22,
                    "min": 5,
                    "max": 4102,
                    "step": 17,
                    "tooltip": "Number of frames to take. Must follow the "
                               "H3 keyframe grid 17k+5 (5, 22, 39...). "
                               "Output images are snapped to the same grid."}),
                "from_mode": (["beginning", "end"], {
                    "default": "beginning",
                    "tooltip": "beginning takes the first from_frames; "
                               "end takes the last from_frames."}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Source video frames. Leave unwired to skip "
                               "picture and only extract audio."}),
                "audio": ("AUDIO", {
                    "tooltip": "Source audio for the same clip. Sliced by "
                               "the matching duration so sound stays locked "
                               "to the taken frames. Leave unwired for "
                               "silent clips."}),
                "ref_image": ("IMAGE", {
                                    "tooltip": "角色参考图"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 240.0,
                    "step": 0.001,
                    "tooltip": "Frame rate used to convert from_frames into "
                               "an audio duration. Must match the source clip."}),
                "keep_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When True, slice the wired audio to match "
                               "the taken frames. When False, skip audio "
                               "and return None even if audio is wired."}),
                "add_noise": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When True, mix 50% Gaussian sampling noise "
                               "into output frames from wired images."}),
                "idx_mode": (["first_frame", "last_frame"], {
                                    "default": "first_frame",
                                    "tooltip": "The frame_idx added to the downstream first frame is first_frame, "
                                    "and the frame_idx added to the last frame is last_frame."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT")
    RETURN_NAMES = ("images", "audio", "count", "frame_idx")
    FUNCTION = "extract"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Take the first or last 17k+5 frames from optional video and audio, "
        "keeping picture and sound locked to the same duration. count is "
        "the kept frame count; frame_idx is +count in beginning mode and "
        "-count in end mode.")

    def extract(self, from_frames=22, from_mode="beginning", images=None,
                audio=None, 
                ref_image=None, 
                fps=24.0, keep_audio=True,
                add_noise=False, idx_mode="first_frame",
                ):
        """按 mode 从片头或片尾取出对齐后的图像与音频。"""
        has_source_images = _image_count(images) > 0
        requested = _snap_h3_frames(from_frames)
        if requested != int(from_frames):
            _LOG.warning(
                "h3_video_from: snapped from_frames %d to H3 grid %d",
                int(from_frames), requested)
        from_end = str(from_mode).strip().lower() == "end"
        keep = requested

        # 未接入图像或图像为空时，使用参考图的第一帧补足请求的帧数。
        if _image_count(images) == 0 and _image_count(ref_image) > 0:
            images = ref_image[:1].expand(requested, *ref_image.shape[1:]).clone()

        out_images = None
        if images is not None:
            total = int(images.shape[0])
            if total <= 0:
                # 空张量按未接入图像处理，保持空结果统一为 None。
                images = None
                keep = 0
            else:
                keep = _resolve_keep(requested, total)
                if total < keep:
                    # 不足最小栅格 5 帧时重复末帧补齐，保证返回仍是 17k+5
                    out_images = _pad_images(images, keep)
                    _LOG.warning(
                        "h3_video_from: padded %d frames to %d so output "
                        "stays on the H3 17k+5 grid",
                        total, keep)
                elif from_end:
                    out_images = images[-keep:]
                else:
                    out_images = images[:keep]
                if keep != requested:
                    _LOG.warning(
                        "h3_video_from: reduced from_frames %d to %d to fit "
                        "a %d frame clip",
                        requested, keep, total)

        # 参考图只替换一个位置，避免改变输出帧数，也不修改输入图像。
        if out_images is not None and _image_count(out_images) > 0 and ref_image is not None:
            target_height = int(out_images.shape[1])
            target_width = int(out_images.shape[2])
            resized_ref = _resize_ref_image(ref_image, target_height, target_width)
            if resized_ref is not None:
                out_images = out_images.clone()
                replace_index = 0 if from_end else -1
                out_images[replace_index] = resized_ref.to(
                    device=out_images.device,
                    dtype=out_images.dtype,
                )

        # 仅对实际接入且非空的 images 加噪；没有 images 而由 ref_image 补帧时，
        # 所有输出帧必须保持清晰。加噪用于模拟采样尚未完成的图像状态，以降低过曝风险。
        # 若 ref_image 有效，exclude_index 会在最终混合后恢复其替换帧原图；此规则优先于上述强度序列。
        if add_noise and has_source_images and out_images is not None:
            frame_count = int(out_images.shape[0])
            # 内部可调的清晰帧数量；设置为 1 或 2 等整数即可调整保留帧数。
            clear_frame_count = 1
            clear_frame_count = min(frame_count, max(0, int(clear_frame_count)))
            gradient_frame_count = frame_count - clear_frame_count * 2
            if gradient_frame_count > 0:
                # 关闭线性模式时，每一帧使用相同的最小噪声强度。
                # sampling_noise_strength = 0.4
                noise_center = 0.50 # 中间
                # 邻近拼接点的起始帧
                # 如果加噪, 拼接处可能有未去除的噪点, 暂定 = 0 不加噪
                noise_near = 0.00  
                if from_end:
                    # [首0.1][...noise_center...][尾0.2]
                    sampling_noise_strength = torch.cat([
                        torch.ones(
                            clear_frame_count, dtype=out_images.dtype,
                            device=out_images.device) * 0.2,
                        torch.linspace(
                            noise_center, noise_center,
                            gradient_frame_count, dtype=out_images.dtype,
                            device=out_images.device),
                        torch.ones(
                            clear_frame_count, dtype=out_images.dtype,
                            device=out_images.device) * noise_near,
                    ])
                else:
                    # [首0.2][..noise_center...][尾0.1]
                    sampling_noise_strength = torch.cat([
                        torch.ones(
                            clear_frame_count, dtype=out_images.dtype,
                            device=out_images.device) * noise_near,
                        torch.linspace(
                            noise_center, noise_center,
                            gradient_frame_count, dtype=out_images.dtype,
                            device=out_images.device),
                        torch.ones(
                            clear_frame_count, dtype=out_images.dtype,
                            device=out_images.device) * 0.2,
                    ])
            else:
                # 防止手动配置为负数、浮点数或超过总帧数时产生非法强度序列。
                sampling_noise_strength = torch.zeros(
                                frame_count, dtype=out_images.dtype,
                                device=out_images.device)

            # 将最终强度序列的中间帧设置为 0.0，形成清晰的中间位置。
            # 例如 22 帧时 middle_index = 22 // 2 = 11，即第 12 帧（Python 从 0 开始计数）。
            # 同时尝试清除 middle_index + 1；奇数帧时会自动跳过越界索引。
            strength_count = sampling_noise_strength.numel()
            if strength_count > 0:
                noise_strength = 0.05 # 不能是 0，否则后续K采样有亮度差（0.5）
                # middle_index
                middle_index = int( strength_count // 2 )
                if from_end:
                    # 从 1 到 middle_index（不包含开始第一个）
                    sampling_noise_strength[:middle_index + 1] = noise_strength
                else:
                    # 从 middle_index 到倒数第二个元素（不包含最后一个）
                    sampling_noise_strength[middle_index:-1] = noise_strength

            noise_exclude_index = None
            if _image_count(ref_image) > 0:
                noise_exclude_index = 0 if from_end else -1
            out_images = _add_sampling_noise(
                out_images, noise_strength=sampling_noise_strength,
                exclude_index=noise_exclude_index)

        out_audio = None
        # keep_audio=False 时跳过音频处理，即使接入了 audio 也返回 None
        if keep_audio and audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            length = int(waveform.shape[-1])
            if images is None:
                # 仅音频时，按波形能覆盖的最大 17k+5 帧决定取出时长
                audio_frames = int(round(length / float(sr) * float(fps)))
                keep = _resolve_keep(requested, audio_frames)
                if keep != requested:
                    _LOG.warning(
                        "h3_video_from: reduced from_frames %d to %d to fit "
                        "%.3fs of audio",
                        requested, keep, length / float(sr))
            sliced = _slice_audio(waveform, sr, fps, keep, from_end)
            out_audio = {"waveform": sliced, "sample_rate": sr}
            _LOG.info(
                "h3_video_from: mode=%s keep=%d frames / %.4fs picture, "
                "%.4fs sound",
                "end" if from_end else "beginning", keep,
                keep / float(fps) if keep else 0.0,
                int(sliced.shape[-1]) / float(sr))
        elif out_images is not None and keep:
            _LOG.info(
                "h3_video_from: mode=%s took %d frames, audio %s",
                "end" if from_end else "beginning", keep,
                "skipped" if not keep_audio else "unwired")

        # count 为实际取出的正整数帧数；无图像时为 0
        count = _image_count(out_images)
        # frame_idx 值主要提供给下游节点 MiniMaxH3AddGuide 用于设置帧 idx 索引位置
        # first_frame 首帧: 返回正值（注意: 首帧始终从0开始, 固定返回0），例如: 0
        # last_frame 尾帧: 返回负值（片尾-count）,例如: -22
        idx_mode_first = str(idx_mode).strip().lower() == "first_frame"
        frame_idx = 0 if idx_mode_first else -count
        # 多裁掉几帧 ?
        cut_count = count
        # cut_count = max(0, cut_count + 5)
        return (out_images, out_audio, cut_count, frame_idx)


NODE_CLASS_MAPPINGS = {
    "H3VideoAudioFrom": H3VideoAudioFrom,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VideoAudioFrom": "H3 Video Audio From",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}