import torch
import logging
_LOG = logging.getLogger("H3VideoTrim")


# ==========================================================================
# 主要用于在支持 MiniMaxH3AddGuide 方便首尾帧控制视频生成后的对首尾进行裁剪
# ==========================================================================


def _clamp_span(total, head, tail):
    """将头尾裁剪量钳制到合法范围，至少保留 1 个元素。

    优先满足 head（开头裁剪），再从剩余长度中裁 tail。
    即使输入超过视频或音频本身长度，也不会出现越界报错。
    total <= 0 时返回 (0, 0, 0)。
    返回 (head, tail, keep)。
    """
    total = int(total)
    head = max(0, int(head))
    tail = max(0, int(tail))
    if total <= 0:
        return 0, 0, 0
    # 开头裁剪已覆盖整段时，保留最后 1 个元素，不再裁尾
    if head >= total:
        return total - 1, 0, 1
    # 剩余长度中至少留 1 个元素给中间段
    tail = min(tail, total - head - 1)
    return head, tail, total - head - tail


class H3VideoAudioCut:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The same pairing applies at the tail when cut_last_frames is set:
    drop that many picture frames and the matching audio duration so the
    remaining middle stays locked. Values that would eat the whole clip
    are clamped; at least one frame (and one audio sample) is kept.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3,
    so the grid rarely lands on a frame boundary. It rounds to the
    NEAREST step, which means a clip ships either about 8.3 ms more sound
    than picture or about 8.3 ms less, depending on its length:

        frames % 3 == 0   exact
        frames % 3 == 1   124 wants 206.67 steps, gets 207, sound is long
        frames % 3 == 2   260 wants 433.33 steps, gets 433, sound is short

    Either way the error compounds. Concatenate two clips and the second
    seam is out by 16.7 ms, three and it is 25 ms, and it grows without
    bound down a chain. It reads as a faint dampening at the first join
    and a short click at later ones. Matching the tail to exactly
    frames/fps stops it accumulating: a long tail is truncated, a short
    one is zero-padded. The padded samples are sound the model never
    generated, so silence is the only honest fill.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：图像、开头裁剪帧数，以及可选的尾部裁剪与音频对齐参数。"""
        return {
            "required": {
                "images": ("IMAGE",),
                "cut_start_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "cut_last_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Number of frames to drop from the tail. "
                               "0 leaves the end unchanged. Audio is "
                               "trimmed by the matching duration. Values "
                               "beyond the remaining length are clamped "
                               "so at least one frame is kept."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Make the audio duration equal frames/fps "
                               "exactly, trimming a long tail or padding a "
                               "short one with silence. H3 rounds its audio "
                               "grid to the nearest step, so each clip "
                               "carries about 8ms too much or too little "
                               "sound, which accumulates at every join in a "
                               "chain."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Remove leading and optional trailing frames from a "
                   "decoded H3 clip, trimming picture and sound by the "
                   "same duration.")

    def trim(self, images, cut_start_frames, audio=None, fps=24.0, match_tail=True,
             cut_last_frames=0):
        """按头尾帧数裁剪视频图像，并按同样时长裁剪对齐音频。"""
        requested_head = max(0, int(cut_start_frames))
        requested_tail = max(0, int(cut_last_frames))
        total = int(images.shape[0])
        # 钳制裁剪范围：完整输入 = 开头裁掉 + 中间返回 + 尾部裁掉
        n, n_last, frames_left = _clamp_span(
            total, requested_head, requested_tail)
        if (n, n_last) != (requested_head, requested_tail):
            _LOG.warning(
                "h3_motion_context: clamped cut_start_frames %d / "
                "cut_last_frames %d to %d / %d on a %d frame clip "
                "(keeping %d frames)",
                requested_head, requested_tail, n, n_last, total, frames_left)
        end = total - n_last
        out_images = images[n:end] if (n or n_last) else images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            last_seconds = n_last / float(fps)
            cut = int(round(seconds * sr))
            cut_last = int(round(last_seconds * sr))
            length = int(waveform.shape[-1])
            # 音频头尾裁剪同样钳制，避免超过波形长度
            cut, cut_last, _kept = _clamp_span(length, cut, cut_last)
            end_sample = length - cut_last
            waveform = waveform[..., cut:end_sample]

            if match_tail:
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    # H3 rounds to the nearest audio step, so a third of
                    # clip lengths ship slightly LESS sound than picture
                    # rather than more. The missing samples are sound
                    # that was never generated, so zero is the honest
                    # fill; anything else would fabricate or attenuate
                    # real content to hide a seam. Leaving it short
                    # instead drifts every later clip earlier, and unlike
                    # the long case that error compounds down the chain.
                    # This also restores what the vae path assumes when
                    # it sets overhang to 0.
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform,
                                                       (0, missing))
                    _LOG.info("h3_motion_context: tail padded %d zero "
                              "samples (%.2fms) so audio matches %d "
                              "frames exactly",
                              missing, missing / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      frames_left, frames_left / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs(frames_left / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n or n_last:
            _LOG.info("h3_motion_context: trimmed %d leading and %d trailing "
                      "frames, %d remain. No audio wired; if this clip has "
                      "sound, mux it through this node or it will drift "
                      "relative to the picture.",
                      n, n_last, frames_left)

        return (out_images, out_audio)


NODE_CLASS_MAPPINGS = {
    "H3VideoAudioCut": H3VideoAudioCut,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3VideoAudioCut": "H3 Video Audio Cut",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
