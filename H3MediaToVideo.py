import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import ComfyExtension, io

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


# 参考图尺寸下拉选项：match/max 保持原语义，数字为预设长边
REF_IMAGE_SIZE_OPTIONS = ["match", "864", "1056", "1280", "1536", "1920", "max"]

# --------------------------------------------------------------
# 相同参数下，不同 ref_image_size 尺寸的采样速度(640-8秒)：
# 参考图的分辨率越高，采样速度明显越慢，建议选择 1280 左右。
# --------------------------------------------------------------
# match : 9s/it （64s） <-- 抽卡用
# 864  : 14s/it （61s） <-- 抽卡用
# 1056 : 15s/it （91s）
# 1280 : 16s/it （99s） <-- 生成用
# 1536 : 20s/it （150s）<-- 生成用
# --------------------------------------------------------------

# 特别说明： 对于 H3 视频生成最小尺寸必须 >= 640, 否则多参会不正确。


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)

def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)

def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _align_dim(value):
    """将尺寸对齐到 CANVAS_MULTIPLE，且不小于该倍数。"""
    return max(CANVAS_MULTIPLE, round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _calc_ref_image_target_size(img_w, img_h, ref_image_size, gen_width, gen_height):
    """按 ref_image_size 计算参考图目标宽高，结果对齐到 32 像素。"""
    if ref_image_size == "match":
        # 按生成画布像素面积等比缩小（不放大）
        scale = min(1.0, math.sqrt((gen_width * gen_height) / (img_w * img_h)))
    elif ref_image_size == "max":
        # 按参考管线 2048 短边上限等比缩小（不放大）
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(img_w, img_h))
    else:
        # 预设长边上限：仅当最长边超过该值时等比缩小，不放大
        target_long = int(ref_image_size)
        scale = min(1.0, target_long / max(img_w, img_h))
    return _align_dim(img_w * scale), _align_dim(img_h * scale)
    
    
def _encode_ref_audio(audio_vae, audio):
    waveform = audio["waveform"]  # [B, C, L]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, z.shape[-1]


def _cached_encode_ref_audio(audio_vae, audio, cache, key):
    """按 key 缓存参考音频编码，避免 low/high 两路重复调用音频 VAE。"""
    if key not in cache:
        cache[key] = _encode_ref_audio(audio_vae, audio)
    return cache[key]


def _build_reference_payload(vae, audio_vae, width, height, frame_count, ref_image_size,
                             ref_images, ref_videos, ref_video_audios, ref_audios,
                             audio_cache):
    """按指定 ref_image_size 构建 tokenizer 展示项与 DiT 参考块。

    图像与视频按该尺寸缩放并编码；音频 latent 通过 audio_cache 复用，不受尺寸影响。
    """
    ref_items = []   # for the tokenizer presentation, in request order
    ref_blocks = []  # for the DiT payload, same order

    for img in (ref_images or {}).values():
        if img is None:
            continue
        h, w = img.shape[1], img.shape[2]
        # 修改匹配:
        tw, th = _calc_ref_image_target_size(w, h, ref_image_size, width, height)
        resized = _resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
        audio_key = name.rsplit("_", 1)[-1]
        soundtrack = ref_video_audios.get("ref_video_audio_" + audio_key)
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        # 修改匹配:
        cw, ch = _calc_ref_image_target_size(vw, vh, ref_image_size, width, height)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = _cached_encode_ref_audio(
                audio_vae, soundtrack, audio_cache, ("video_audio", audio_key))
            # the soundtrack gets its own <Audio j> label, emitted before <Video k>
            ref_items.append({"type": "audio"})
        # Qwen sees the video at 2 fps with timestamps
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        qwen_frames = frames[sample_idx]
        ref_items.append({"type": "video", "data": qwen_frames,
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for name, audio in (ref_audios or {}).items():
        if audio is None:
            continue
        audio_latent, ref_audio_t = _cached_encode_ref_audio(
            audio_vae, audio, audio_cache, ("audio", name))
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

    return ref_items, ref_blocks


def _encode_positive(clip, vae, audio_vae, prompt, width, height, frame_count,
                     ref_image_size, ref_images, ref_videos, ref_video_audios,
                     ref_audios, audio_cache):
    """按指定参考尺寸编码 positive conditioning，音频块来自共享缓存。"""
    ref_items, ref_blocks = _build_reference_payload(
        vae, audio_vae, width, height, frame_count, ref_image_size,
        ref_images, ref_videos, ref_video_audios, ref_audios, audio_cache)
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    return cond


def _normalize_media_input(value):
    """将 IMAGE Tensor 包装为标准 H3_MEDIA，并保留已有媒体对象。"""
    if isinstance(value, dict) and any(
            key in value for key in ("type", "image", "audio", "role_name", "prompt")):
        return value
    if isinstance(value, torch.Tensor):
        if len(value.shape) != 4 or value.shape[-1] not in (1, 3, 4):
            raise TypeError(
                "H3MediaPrompt: IMAGE input must be a [B, H, W, C] Tensor "
                "with 1, 3, or 4 channels"
            )
        return {
            "type": "IMAGE",
            "image": value,
            "audio": None,
            "role_name": None,
            "prompt": None,
        }
    raise TypeError("H3MediaPrompt: media input must be an H3_MEDIA object or IMAGE Tensor")


def _build_medias_passthrough(medias=None, media_slots=None):
    """稳定排序并透传媒体对象，将裸 IMAGE 包装后排列在 H3_MEDIA 末尾。"""
    h3_media_values = []
    image_media_values = []

    def append_media_values(value):
        """递归展开媒体集合，并按原始输入类型分别收集媒体。"""
        if value is None:
            return
        if isinstance(value, dict) and any(
                key in value for key in ("type", "image", "audio", "role_name", "prompt")):
            h3_media_values.append(value)
        elif isinstance(value, dict):
            for nested_value in value.values():
                append_media_values(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                append_media_values(nested_value)
        else:
            image_media_values.append(_normalize_media_input(value))

    append_media_values(medias)
    for index in range(1, 10):
        append_media_values((media_slots or {}).get(f"media_{index}"))

    media_values = h3_media_values + image_media_values
    if not media_values:
        return None
    if len(media_values) == 1:
        return media_values[0]
    return {
        f"media_{index}": media
        for index, media in enumerate(media_values, start=1)
    }


def _collect_medias(medias=None):
    """从已排序并标准化的透传结果中收集参考素材与媒体组。"""
    ref_images = {}
    ref_audios = {}
    items = []

    if medias is None:
        media_values = []
    elif isinstance(medias, dict) and any(
            key in medias for key in ("type", "image", "audio", "role_name", "prompt")):
        media_values = [medias]
    elif isinstance(medias, dict):
        media_values = list(medias.values())
    else:
        raise TypeError("H3MediaPrompt: preprocessed medias must be an H3_MEDIA object or mapping")

    for index, media in enumerate(media_values, start=1):
        if not isinstance(media, dict) or not any(
                key in media for key in ("image", "audio", "role_name", "prompt")):
            raise TypeError("H3MediaPrompt: each preprocessed media must be an H3_MEDIA object")
        item = {
            "type": media.get("type"),
            "image": media.get("image"),
            "audio": media.get("audio"),
            "role_name": media.get("role_name"),
            "prompt": media.get("prompt"),
        }
        items.append(item)
        if item["image"] is not None:
            ref_images[f"media_image_{index}"] = item["image"]
        if item["audio"] is not None:
            ref_audios[f"media_audio_{index}"] = item["audio"]

    return ref_images, ref_audios, items



class H3MediaToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        """定义双参考尺寸输入与 positive_low / positive_high 双路输出。"""
        return io.Schema(
            node_id="H3MediaToVideo",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting. subject_definitions describes connected image subjects and their audio references.",
            display_name="H3 Media To Video (Reference)",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size_low", options=REF_IMAGE_SIZE_OPTIONS, default="match",
                    tooltip="Low-res reference image sizing for positive_low. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 1056/1280/1536/1920 cap the long edge to that size (down only, short edge follows aspect, 32-aligned); 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so larger sizes are slower."),
                io.Combo.Input("ref_image_size_high", options=REF_IMAGE_SIZE_OPTIONS, default="1280",
                    tooltip="High-res reference image sizing for positive_high. Options and downscale rules match ref_image_size_low. Use a larger preset here for the second sampling stage."),
                io.Custom("H3_MEDIA").Input("medias", optional=True,
                    tooltip="Optional media from H3MediaLoader. The single port accepts multiple connections."),
                io.Custom("H3_MEDIA").Input("media_1", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_2", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_3", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_4", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_5", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_6", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_7", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_8", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA").Input("media_9", optional=True, extra_dict={"hidden": True}),
                io.Autogrow.Input("ref_video_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video_image", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_image_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="positive_high"),
                io.Latent.Output(),
                io.String.Output(display_name="prompt_info"),
                ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size_low="match", ref_image_size_high="1280",
                medias=None, ref_video_images=None, ref_video_audios=None, **kwargs) -> io.NodeOutput:
        """收集媒体组、编码 conditioning，并输出角色主体定义文本。"""
        latent, frame_count = _empty_av_latent(width, height, length)
        media_slots = {name: kwargs.get(name) for name in (
            "media_1", "media_2", "media_3", "media_4", "media_5",
            "media_6", "media_7", "media_8", "media_9")}
        
        # 标准化媒体并稳定排序：H3_MEDIA 在前，裸 IMAGE 包装后排列在末尾。
        passthrough_medias = _build_medias_passthrough(medias, media_slots)
        # 直接从已处理的透传结果中收集多参素材，避免重复标准化和排序。
        ref_images, ref_audios, items = _collect_medias(passthrough_medias)
        
        # 提示词处理功能由 H3MediaPrompt 负责
        # 将用户输入的对话, 替换为 H3 的规则 <d>[Chinese] ...</d> 标签。
        prompt = PromptPeplace.replace_prompt_dialogues(prompt)
        
        
        # 音频不受尺寸影响，两路共享缓存，避免重复编码
        audio_cache = {}
        encode_kwargs = dict(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
            width=width, height=height, frame_count=frame_count,
            ref_images=ref_images, ref_videos=ref_video_images,
            ref_video_audios=ref_video_audios, ref_audios=ref_audios,
            audio_cache=audio_cache,
        )
        cond_low = _encode_positive(ref_image_size=ref_image_size_low, **encode_kwargs)
        cond_high = _encode_positive(ref_image_size=ref_image_size_high, **encode_kwargs)
        return io.NodeOutput(cond_low, cond_high, latent, prompt)


import re

class PromptPeplace:
    """H3 Prompt Replace：将文本中引号内的对话内容转换为 <d>[Chinese] ...</d> 标签。

    规则：
    - 匹配中英文引号内的全部对话内容
    - 新引号对话替换为 <d>[Chinese] 对话内容</d>
    - 引号本身被去掉，其余字符保持不变
    - 任意已存在的 <d>...</d> 标签视为保护块，不再二次包裹
    - 保护块内部的中英引号、嵌套 <d> 与重复语言标记一律清除
    - 语言标记为方括号内任意语言单词，如 [Chinese] / [English] / [Japanese]
    - 已有标签保留其原有语言标记；无标记时默认 [Chinese]
    - 规范结果始终为单层 <d>[语言] 对话内容</d>
    - 支持的引号对：
    - ASCII 双引号： "..."
    - 中文弯引号： “...”
    - 直角引号： 「...」
    - 双直角引号： 『...』
    - 全角引号： ＂...＂
    """

    # 开引号到闭引号的配对
    _QUOTE_PAIRS = {
        '"': '"',
        '“': '”',
        '「': '」',
        '『': '』',
        '＂': '＂',
        '〝': '〞',
    }

    # 需要从对话正文中清除的中英文引号字符
    _QUOTE_CHARS_RE = re.compile(r'["“”「」『』＂〝〞]')

    # 清除嵌套残留的 <d> / </d> 标签
    _D_TAG_RE = re.compile(r'</?d>', re.IGNORECASE)

    # 语言标记：方括号内的任意语言单词，如 [Chinese] / [English] / [zh-CN]
    _LANGUAGE_MARK_RE = re.compile(r'\[([A-Za-z][A-Za-z0-9_\-]*)\]')

    # 新引号对话默认使用的语言标记
    _DEFAULT_LANGUAGE = "Chinese"


    def _strip_quote_chars(content):
        """清除对话内容中的中英文引号字符，并去掉首尾空白。"""
        return PromptPeplace._QUOTE_CHARS_RE.sub('', content).strip()


    def _flatten_dialogue_inner(content):
        """展开嵌套 d 标签与语言标记，并清除内部引号。

        返回 (语言单词, 清洗后的对话正文)。
        若原文无语言标记，语言回退为 Chinese。
        """
        match = PromptPeplace._LANGUAGE_MARK_RE.search(content)
        language = match.group(1) if match else PromptPeplace._DEFAULT_LANGUAGE
        text = PromptPeplace._D_TAG_RE.sub('', content)
        text = PromptPeplace._LANGUAGE_MARK_RE.sub('', text)
        return language, PromptPeplace._strip_quote_chars(text)


    def _wrap_dialogue(content):
        """将对话正文规范包裹为单层 <d>[语言] 内容</d>。"""
        language, inner = PromptPeplace._flatten_dialogue_inner(content)
        return f"<d>[{language}] {inner}</d>"


    def _starts_open_d(text, index):
        """判断 index 处是否为 <d> 开标签。"""
        return text[index:index + 3].lower() == '<d>'


    def _starts_close_d(text, index):
        """判断 index 处是否为 </d> 闭标签。"""
        return text[index:index + 4].lower() == '</d>'


    def _find_matching_close_d(text, start):
        """从 <d> 起始位置查找配对的 </d> 起始下标；未闭合返回 -1。"""
        depth = 1
        index = start + 3
        length = len(text)
        while index < length:
            if PromptPeplace._starts_close_d(text, index):
                depth -= 1
                if depth == 0:
                    return index
                index += 4
                continue
            if PromptPeplace._starts_open_d(text, index):
                depth += 1
                index += 3
                continue
            index += 1
        return -1


    def replace_prompt_dialogues(text):
        """把输入文本里所有引号内对话内容替换为对话标签。

        已存在的任意 <d>...</d> 保护块不再二次包裹，但会清除内部引号并展平嵌套。
        保护块会保留已有语言标记（任意语言单词）；新引号对话默认使用 [Chinese]。
        其它字符原样保留；text 为 None 时按空字符串处理。
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        result = []
        index = 0
        length = len(text)
        while index < length:
            if PromptPeplace._starts_open_d(text, index):
                close_index = PromptPeplace._find_matching_close_d(text, index)
                if close_index != -1:
                    result.append(PromptPeplace._wrap_dialogue(text[index + 3:close_index]))
                    index = close_index + 4
                    continue
            open_quote = text[index]
            close_quote = PromptPeplace._QUOTE_PAIRS.get(open_quote)
            if close_quote is not None:
                close_index = text.find(close_quote, index + 1)
                if close_index != -1:
                    result.append(PromptPeplace._wrap_dialogue(text[index + 1:close_index]))
                    index = close_index + 1
                    continue
            result.append(text[index])
            index += 1
        return ''.join(result)




NODE_CLASS_MAPPINGS = {
    "H3MediaToVideo": H3MediaToVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MediaToVideo": "H3 Media To Video (Reference)",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
