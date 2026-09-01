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


def _build_subject_definitions(items):
    """按媒体组实际输入顺序拼接角色主体定义文本。
    """
    audio_lines = []
    # 索引计数器
    subject_index = 0
    audio_index = 0
    # 需要收集角色名称对应的Subject,用于后续处理
    role_names_map = {}
    # 每个组里的 图像 和 音频 都会按顺序上传到模型参考中 ref_images, ref_audios
    subject_lines = []
    # 所以,它们的索引计数序号需要分别从 1 开始
    for item in items or []:
        
        # 如果是纯 IMAGE 对象接入, 则仅参考图像, 无需提示词绑定
        item_type = item.get("type")
        if item_type is not None and item_type != "H3_MEDIA":
            # 跳过非 H3_MEDIA 对象的提示词处理
            continue
        
        # image 需要从 1 开始, 按上传顺序
        has_image = False
        if item.get("image") is not None:
            has_image = True
            subject_index += 1 # 正常计数
            subject_tag = f"<Subject {subject_index}>(S{subject_index})"
            subject_parts = [
                value for value in (item.get("role_name"), item.get("prompt"))
                if value is not None and str(value) != ""
            ]
            subject_text = " - ".join(str(value) for value in subject_parts)
            subject_lines.append(f"{subject_tag}: {subject_text}")
            
            # 把 role_name 关联 subject_tag
            role_name = item.get("role_name")
            if role_name  is not None and str(role_name).strip() != "":
                role_name = role_name.strip()
                role_names_map[role_name] = subject_tag
                
        # <Audio N> 使用以下关系标记：
        # fully_copy 完整源音频作为目标视频的完整最终音轨
        # reference 不直接复制信号，只参考音色、节奏、音乐风格、台词内容或声音质感
            
        # audio 需要从 1 开始, 按上传顺序
        if item.get("audio") is not None:
            audio_index += 1 # 正常计数
            # 但前提需要有图像, 才需要拼这个提示词
            if has_image:
                # 有角色参考时, 让音频是说话人音色参考
                # audio_lines.append(
                #     f"<Audio {audio_index}> 是 {subject_tag} 的说话音色风格参考,"
                #     "不复制原始音频信号。"
                # )
                audio_lines.append(
                    f"<Audio {audio_index}> : reference - its vocal timbre guides the dialogue delivery of  {subject_tag},"
                    "without copying the original signal."
                )
            else:
                # 未提供角色时, 让音频参考其节奏或节拍, 当音乐用.
                # audio_lines.append(
                #     f"<Audio {audio_index}> 参考节拍、节奏、音乐风格或声音连续性。"
                # )
                audio_lines.append(
                    f"<Audio {audio_index}> fully_copy - Refer to beats, rhythms, musical styles, or sound continuity."
                )
    
    # 若有则标签头
    if len(subject_lines) > 0:
        subject_lines.insert(0, "subject_definitions:")              
    
    # 根据顺序组合
    lines = subject_lines + audio_lines
    
    return "\n".join(lines) + "\n", role_names_map, 



class H3MediaPrompt(io.ComfyNode):
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
            node_id="H3MediaPrompt",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting. subject_definitions describes connected image subjects and their audio references.",
            display_name="H3 Media Prompt (Reference)",
            category="model/conditioning/minimax",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Custom("H3_MEDIA,IMAGE").Input("medias", optional=True,
                    tooltip="Optional H3_MEDIA or IMAGE input. The single port accepts multiple connections."),
                io.Custom("H3_MEDIA,IMAGE").Input("media_1", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_2", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_3", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_4", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_5", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_6", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_7", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_8", optional=True, extra_dict={"hidden": True}),
                io.Custom("H3_MEDIA,IMAGE").Input("media_9", optional=True, extra_dict={"hidden": True}),
            ],
            outputs=[
                io.Custom("H3_MEDIA").Output("medias"),
                io.String.Output(display_name="prompt"),
                ],
        )

    @classmethod
    def execute(cls,  prompt, medias=None, **kwargs) -> io.NodeOutput:
        """收集媒体组、编码 conditioning，并输出角色主体定义文本。"""
        media_slots = {name: kwargs.get(name) for name in (
            "media_1", "media_2", "media_3", "media_4", "media_5",
            "media_6", "media_7", "media_8", "media_9")}

        # 标准化媒体并稳定排序：H3_MEDIA 在前，裸 IMAGE 包装后排列在末尾。
        passthrough_medias = _build_medias_passthrough(medias, media_slots)
        # 直接从已处理的透传结果中收集多参素材，避免重复标准化和排序。
        ref_images, ref_audios, items = _collect_medias(passthrough_medias)
        # 生成多参提示
        subject_definitions, role_names_map = _build_subject_definitions(items)
        
        # --------------------------------------------------------------
        # minimax h3 的提示词, 采用 6 段式结构, 这里只用到 5 段足够;
        # --------------------------------------------------------------
        # 以下 2 段由输入参考素材(图像/音频)得到:
        # subject_definitions:
        # summary:
        # --------------------------------------------------------------
        # 以下 3 段由 prompt 用户输入文本提示获取:
        # integrated_multimodal_description: (必需 * ) 
        # overall_soundscape: (可自动追加)
        # non_diegetic_music: (可自动追加) 
        # --------------------------------------------------------------
        
        # 若用户没有输入 summary: 则自动追加说明信息:
        if "summary:".lower() not in prompt.lower():
            summary = "\n"
            summary += "summary:\n"
            summary += "[reference generation + reference generation]\n"
            # summary += "手持镜头\n"
            prompt = f"{summary}{prompt}"
        
        
        # 若用户没有输入 integrated_multimodal_description 或 detailed_description 则自动追加:
        if ("detailed_description".lower() not in prompt.lower()) and \
            ("integrated_multimodal_description".lower() not in prompt.lower()):
            # 两种标签二选一, 多参建议选择 detailed_description 标签
            prompt = f"\ndetailed_description:\n{prompt}" # 用于多参模式 (必需 * )
            # prompt = f"\nintegrated_multimodal_description:\n{prompt}" # 用于T2V模式 (必需 * )
        
        # 将用户输入的对话, 替换为 H3 的规则 <d>[Chinese] ...</d> 标签。
        prompt = PromptPeplace.replace_prompt_dialogues(prompt)
        
        # 将角色名称加上对应的 subject_tag 关联参考图素材ID和音频ID
        # 比如: 小华 --> 小华<Subject 1>(S1)
        for role_name in role_names_map:
            subject_tag = role_names_map.get(role_name)
            # print(f"角色: {role_name} {subject_tag}")
            if subject_tag is not None:
                prompt = prompt.replace(role_name, f"{role_name}{subject_tag}")
        
        # 核心提示词输入
        prompt = f"{subject_definitions}\n{prompt}\n"
        
        # 若用户没有输入 overall_soundscape 则自动追加:
        if "overall_soundscape".lower() not in prompt.lower():
            prompt += "\n"
            prompt += "overall_soundscape:\n"
            prompt += "N/A\n"
            
        # 若用户没有输入 non_diegetic_music 则自动追加:
        if "non_diegetic_music".lower() not in prompt.lower():
            prompt += "\n"
            prompt += "non_diegetic_music:\n"
            prompt += "N/A\n"
            
        
        # -------------------------------------------------------------
        # 处理完成的最终 prompt 提示词示例, 符合 5 段时结构:
        # -------------------------------------------------------------
        # subject_definitions:
        # <Subject 1>(S1): 小美 - 她是一个女人
        # <Subject 2>(S2): 小华 - 她是一个男人
        # <Audio 1> 是 <Subject 1>(S1) 的说话音色风格参考,不复制原始音频信号。
        # <Audio 2> 是 <Subject 2>(S2) 的说话音色风格参考,不复制原始音频信号。

        # summary:
        # [reference generation + reference generation]
        # 第一人称手持镜头

        # detailed_description:
        # 小华<Subject 2>(S2)正在跑,她对着镜头说:<d>[Chinese] 你是谁啊</d>,小美<Subject 1>(S1)生气,转头就走

        # overall_soundscape:
        # N/A

        # non_diegetic_music:
        # N/A
        # -------------------------------------------------------------
        
        return io.NodeOutput(passthrough_medias, prompt)


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
    "H3MediaPrompt": H3MediaPrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3MediaPrompt": "H3 Media Prompt (Reference)",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
