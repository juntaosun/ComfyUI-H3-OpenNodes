
import math
import torchaudio

import nodes
import node_helpers
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io

# ==========================================================================
# 官方原节点为: MiniMaxH3AddGuide
# 当前节点的优化与改动如下,支持尺寸输入, 可以对二采进行高/低的 positive 返回支持
# 低采时: 原始节点的默认行为, 节点会缩放 image 的尺寸匹配 latent.
# 二采时: 将 image 缩放到目标尺寸, 这样保证二采图像分辨率 positive 信息不会模糊.
# ==========================================================================

def _resize(image, width, height, crop):
    """将图像批次缩放到指定的像素尺寸。"""
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _align_resolution_to_32(width, height):
    """将正数宽高分别向上对齐到 32 的倍数。"""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须都大于 0")
    return ((width + 31) // 32) * 32, ((height + 31) // 32) * 32


def _calculate_resolution_from_long_side(source_width, source_height, long_side):
    """按图像原始比例计算目标尺寸，并将两条边向上对齐到 32 的倍数。"""
    source_width = int(source_width)
    source_height = int(source_height)
    long_side = int(long_side)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("源图像宽度和高度必须都大于 0")
    if long_side <= 0:
        raise ValueError("长边必须大于 0")

    aligned_long_side, _ = _align_resolution_to_32(long_side, 1)
    if source_width >= source_height:
        target_width = aligned_long_side
        target_height = math.ceil(target_width * source_height / source_width)
    else:
        target_height = aligned_long_side
        target_width = math.ceil(target_height * source_width / source_height)
    return _align_resolution_to_32(target_width, target_height)


def _encode_ref_audio(audio_vae, audio):
    """将输入音频编码为 MiniMax H3 音频 latent。"""
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]

class H3AddGuide(io.ComfyNode):
    """Anchor image and/or audio guides at an arbitrary pixel frame of the target video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3AddGuide",
            display_name="Add Guide for H3 (Custom)",
            category="model/conditioning/minimax",
            description="Anchor an image, a short clip, audio, or a clip with its soundtrack at any frame of a MiniMax H3 video. Chain several nodes to anchor several frames.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("positive_scaler", optional=True,
                                      tooltip="上一节点的 positive_scaler；未连接时默认使用 positive。"),
                io.Latent.Input("latent", optional=True),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when an image is connected."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE, needed when an audio is connected."),
                io.Image.Input("image", optional=True, tooltip="Image or video frames to anchor. Multi-frame batches are anchored as a clip and cropped down to the model's valid clip lengths: 5, 22, 39... (17k + 5) frames. Batches shorter than 5 frames use only the first image."),
                io.Audio.Input("audio", optional=True,
                               tooltip="Soundtrack to anchor starting at the same frame index, cropped to the video's remaining duration."),
                io.Int.Input("frame_idx", default=0, min=-9999, max=9999,
                             tooltip="Frame index to anchor the image or the clip's first frame at. Negative values are counted from the end of the video."),
                io.Int.Input("scaler_long_side", default=1056, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Combo.Input("crop", options=["center", "disabled"], optional=True, default="disabled"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="positive_scaler"),
                io.Image.Output(display_name="crop_images"),
                ],
        )

    @classmethod
    def execute(cls, positive, positive_scaler=None,
                latent=None, 
                vae=None, audio_vae=None,
                image=None, audio=None, 
                frame_idx=0, 
                scaler_long_side=None,
                crop="disabled",
                ) -> io.NodeOutput:
        """分别生成两份图像引导 Conditioning，并返回按裁剪选项处理后的图像。"""
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5 or samples.tensors[0].shape[1] != 24:
            raise ValueError("MiniMaxH3AddGuide expects a MiniMax H3 AV latent")
        if image is None and audio is None:
            raise ValueError("MiniMaxH3AddGuide needs an image or an audio to anchor")
        video = samples.tensors[0]
        height = video.shape[3] * 16
        width = video.shape[4] * 16
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))

        guide_frames = 1
        if image is not None:
            if vae is None:
                raise ValueError("anchoring guide frames needs the vae input")
            guide_frames = image.shape[0]
            if guide_frames < 5:
                guide_frames = 1
            else:
                while guide_frames % 17 != 5:
                    guide_frames -= 1

        resolved_frame_index = frame_idx if frame_idx >= 0 else frame_count + frame_idx
        if resolved_frame_index < 0 or resolved_frame_index + guide_frames > frame_count:
            if guide_frames == 1:
                raise ValueError("frame_idx {} is outside the video's {} frames".format(frame_idx, frame_count))
            raise ValueError("a {} frame guide clip at frame_idx {} does not fit in the video's {} frames".format(
                guide_frames, frame_idx, frame_count))

        keyframe = {"resolved_frame_index": resolved_frame_index}
        scaler_keyframe = {"resolved_frame_index": resolved_frame_index}
        crop_images = None
        if image is not None:
            # positive 保持原逻辑：按输入 latent 的画布尺寸缩放图像。
            # 首尾帧,千万不要 crop 裁剪, 用 "disabled" 保持原始的缩放, 
            # 图像裁剪应该由用户在上游节点自行完成. 图像输入是什么,最终出来就是什么.
            frames = _resize(image[:guide_frames], width, height, crop) # "disabled"
            crop_images = frames
            keyframe["latent"] = vae.encode(frames)

            source_height, source_width = image.shape[1], image.shape[2]
            target_width, target_height = _calculate_resolution_from_long_side(
                source_width, source_height, scaler_long_side or width)
            scaler_frames = _resize(
                image[:guide_frames], target_width, target_height, crop) # "disabled"
            scaler_keyframe["latent"] = vae.encode(scaler_frames)

        if audio is not None:
            if audio_vae is None:
                raise ValueError("anchoring guide audio needs the audio_vae input")
            audio_latent, audio_rt = _encode_ref_audio(audio_vae, audio)
            # the streams share one time axis: FRAME_RESCALE per pixel frame, 1.0 per audio latent frame
            max_rt = math.floor(samples.tensors[1].shape[-1] - FRAME_RESCALE * resolved_frame_index)
            if max_rt < 1:
                raise ValueError("frame_idx {} is past the end of the video's audio track".format(frame_idx))
            if audio_rt > max_rt:
                audio_latent = audio_latent[..., :max_rt].clone()
            keyframe["audio_latent"] = audio_latent
            scaler_keyframe["audio_latent"] = audio_latent

        # 未连接时沿用 positive，连接后保留上一节点的高分辨率 guide 链。
        positive_scaler_input = positive_scaler if positive_scaler is not None else positive

        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        positive_output = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes})

        scaler_keyframes = list(positive_scaler_input[0][1].get("minimax_keyframes", []))
        scaler_keyframes.append(scaler_keyframe)
        positive_scaler_output = node_helpers.conditioning_set_values(
            positive_scaler_input, {"minimax_keyframes": scaler_keyframes})
        return io.NodeOutput(positive_output, positive_scaler_output, crop_images)
    
    
NODE_CLASS_MAPPINGS = {
    "H3AddGuide": H3AddGuide,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AddGuide": "Add Guide for H3 (Custom)",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}