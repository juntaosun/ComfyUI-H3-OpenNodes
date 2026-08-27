"""H3 Audio Cut：将音频开头指定时长替换为静音，其余部分保持不变。

输入 AUDIO，按可调节的毫秒参数（默认 200ms）把波形最前面
对应采样点置零；总时长、采样率与声道结构不变。
"""

import torch


def _silence_audio_head(audio, silence_ms):
    """将音频开头 silence_ms 毫秒的波形置零，后段原样保留。

    audio 为 ComfyUI AUDIO 字典：{"waveform": Tensor, "sample_rate": int}。
    waveform 形状一般为 [batch, channels, samples]。
    silence_ms <= 0 时直接返回输入的浅拷贝；超出总长时整段静音。
    返回新的 AUDIO 字典（waveform 为 clone，不修改原对象）。
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError(
            "h3_audio_cut: expected AUDIO dict with 'waveform' and "
            "'sample_rate'")
    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate <= 0:
        raise ValueError(
            "h3_audio_cut: invalid sample_rate %r" % (audio.get("sample_rate"),))
    if waveform is None:
        raise ValueError("h3_audio_cut: waveform is missing")

    # 复制，避免原地改动上游节点缓存
    out_wave = waveform.clone()
    silence_ms = float(silence_ms)
    if silence_ms <= 0:
        out = dict(audio)
        out["waveform"] = out_wave
        out["sample_rate"] = sample_rate
        return out

    # 计算需要静音的采样点数（四舍五入到最近整数）
    n_silence = int(round(sample_rate * silence_ms / 1000.0))
    if n_silence <= 0:
        out = dict(audio)
        out["waveform"] = out_wave
        out["sample_rate"] = sample_rate
        return out

    # 最后一维为时间采样轴
    total = int(out_wave.shape[-1])
    n = min(n_silence, total)
    if n > 0:
        out_wave[..., :n] = 0

    out = dict(audio)
    out["waveform"] = out_wave
    out["sample_rate"] = sample_rate
    return out


class H3AudioMuteStart:
    """将输入音频开头指定毫秒时长替换为静音，总时长不变。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：AUDIO 与开头静音毫秒数。"""
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "待处理的音频。开头指定毫秒将被置零静音，"
                               "后段与总时长保持不变。"}),
                "silence_ms": ("FLOAT", {
                    "default": 300.0,
                    "min": 0.0,
                    "max": 600000.0,
                    "step": 1.0,
                    "tooltip": "从音频开头起替换为静音的时长（毫秒）。"
                               "默认 300 ms；0 表示不处理；超过总长时整段静音。"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "H3/audio"
    DESCRIPTION = (
        "Replace the beginning of an audio waveform with silence for a "
        "configurable duration (default 200 ms). Total length, sample rate "
        "and channel layout are unchanged; only the leading samples are "
        "zeroed.")

    def process(self, audio, silence_ms=300.0):
        """执行开头静音替换，返回处理后的 AUDIO。"""
        return (_silence_audio_head(audio, silence_ms),)


NODE_CLASS_MAPPINGS = {
    "H3AudioMuteStart": H3AudioMuteStart,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioMuteStart": "H3 Audio Mute Start",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}