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

import re


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
    return _QUOTE_CHARS_RE.sub('', content).strip()


def _flatten_dialogue_inner(content):
    """展开嵌套 d 标签与语言标记，并清除内部引号。

    返回 (语言单词, 清洗后的对话正文)。
    若原文无语言标记，语言回退为 Chinese。
    """
    match = _LANGUAGE_MARK_RE.search(content)
    language = match.group(1) if match else _DEFAULT_LANGUAGE
    text = _D_TAG_RE.sub('', content)
    text = _LANGUAGE_MARK_RE.sub('', text)
    return language, _strip_quote_chars(text)


def _wrap_dialogue(content):
    """将对话正文规范包裹为单层 <d>[语言] 内容</d>。"""
    language, inner = _flatten_dialogue_inner(content)
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
        if _starts_close_d(text, index):
            depth -= 1
            if depth == 0:
                return index
            index += 4
            continue
        if _starts_open_d(text, index):
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
        if _starts_open_d(text, index):
            close_index = _find_matching_close_d(text, index)
            if close_index != -1:
                result.append(_wrap_dialogue(text[index + 3:close_index]))
                index = close_index + 4
                continue
        open_quote = text[index]
        close_quote = _QUOTE_PAIRS.get(open_quote)
        if close_quote is not None:
            close_index = text.find(close_quote, index + 1)
            if close_index != -1:
                result.append(_wrap_dialogue(text[index + 1:close_index]))
                index = close_index + 1
                continue
        result.append(text[index])
        index += 1
    return ''.join(result)


class H3PromptPeplace:
    """将文本输入中引号包裹的对话内容转换为 <d>[Chinese] 对话内容</d>。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：单个多行文本。"""
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "待处理的提示词文本。"
                               "中英文引号内的对话内容将被替换为 "
                               "<d>[Chinese] 对话内容</d>。"
                               "已有 <d>[任意语言] ...</d> 标签会保留语言标记。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "H3/text"
    DESCRIPTION = (
        "Replace all dialogue content inside Chinese/English quotation marks "
        "with <d>[Chinese] dialogue</d>. Existing <d>[Language] ...</d> tags "
        "keep their language marker and are never nested.")

    def process(self, text):
        """执行引号对话内容替换，返回处理后的字符串。"""
        return (replace_prompt_dialogues(text),)


NODE_CLASS_MAPPINGS = {
    "H3PromptPeplace": H3PromptPeplace,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptPeplace": "H3 Prompt Peplace",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
