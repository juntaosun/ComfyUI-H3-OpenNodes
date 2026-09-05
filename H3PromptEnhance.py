"""H3 Prompt Enhance：通过本地 OpenAI 兼容模型服务增强提示词。"""

import json as _json
import urllib.error as _urllib_error
import urllib.request as _urllib_request
import warnings as _warnings
from pathlib import Path as _Path


_REQUEST_TIMEOUT = 180
_MAX_LENGTH = 1024
_NONE_MODEL = "None"
_CUSTOM_SKILL = "Custom"
_SKILLS_DIR = _Path(__file__).resolve().parent / "skills"


def _get_skill_names():
    """扫描插件 skills 目录并返回 Custom 与 Markdown Skill 文件名。"""
    skill_names = [_CUSTOM_SKILL]
    try:
        files = sorted(
            path.name for path in _SKILLS_DIR.glob("*.md")
            if path.is_file())
    except OSError as exc:
        raise RuntimeError(f"无法读取 skills 目录：{exc}") from exc
    return skill_names + files


def _read_system_skill(system_skill, system_prompt):
    """按选择读取 Skill 内容，Custom 模式使用用户输入的系统提示词。"""
    selected_skill = str(system_skill or _CUSTOM_SKILL).strip()
    if not selected_skill or selected_skill.lower() == _CUSTOM_SKILL.lower():
        return str(system_prompt or "")
    if (selected_skill != _Path(selected_skill).name
            or not selected_skill.lower().endswith(".md")):
        raise RuntimeError("无效的 system_skill 文件名")

    skill_path = (_SKILLS_DIR / selected_skill).resolve()
    try:
        skill_path.relative_to(_SKILLS_DIR.resolve())
    except ValueError as exc:
        raise RuntimeError("system_skill 文件必须位于 skills 目录内") from exc
    if not skill_path.is_file():
        raise RuntimeError(f"找不到 system_skill 文件：{selected_skill}")
    try:
        return skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"无法读取 system_skill 文件：{selected_skill}") from exc


def _get_api_base_url(base_url):
    """规范模型服务地址，并移除可能包含的具体接口路径。"""
    normalized_url = str(base_url or "").strip().rstrip("/")
    if not normalized_url:
        raise RuntimeError("模型服务连接地址不能为空")
    for suffix in ("/chat/completions", "/models/unload", "/models"):
        if normalized_url.endswith(suffix):
            return normalized_url[:-len(suffix)]
    return normalized_url


def _build_chat_completions_url(base_url):
    """根据基础地址生成 OpenAI Chat Completions 接口地址。"""
    return _get_api_base_url(base_url) + "/chat/completions"


def _build_models_url(base_url):
    """根据基础地址生成 OpenAI Models 接口地址。"""
    return _get_api_base_url(base_url) + "/models"


def _get_management_base_url(base_url):
    """从 OpenAI API 地址提取 llama.cpp 管理接口的服务根地址。"""
    api_base_url = _get_api_base_url(base_url)
    if api_base_url.endswith("/v1"):
        return api_base_url[:-len("/v1")]
    return api_base_url


def _build_unload_url(base_url):
    """生成 llama.cpp 路由器的模型卸载接口地址。"""
    return _get_management_base_url(base_url) + "/models/unload"


def _build_headers(api_key):
    """创建 JSON 请求头，并仅在 API Key 非空时添加鉴权信息。"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _extract_response_text(data):
    """从 OpenAI 兼容响应中提取并清理助手回复文本。"""
    if not isinstance(data, dict):
        raise RuntimeError("模型服务返回了无效的 JSON 数据结构")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("模型服务未返回有效的提示词")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("模型服务未返回有效的提示词")

    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    else:
        text = choice.get("text", "")

    if isinstance(text, str) and text.strip():
        return text.strip()
    raise RuntimeError("模型服务未返回有效的提示词")


def _extract_model_names(data):
    """从 OpenAI 模型列表响应中提取模型 ID，并按原顺序去重。"""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError("模型服务返回了无效的模型列表")

    names = []
    seen = set()
    for item in data["data"]:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if model_id and model_id not in seen:
            names.append(model_id)
            seen.add(model_id)

    if not names:
        raise RuntimeError("模型服务未返回可用的模型列表")
    return names


def _read_http_error(error):
    """读取 HTTP 错误正文并限制长度，避免错误信息过大。"""
    try:
        detail = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    if detail:
        return f"模型服务请求失败（HTTP {error.code}）：{detail[:1000]}"
    return f"模型服务请求失败（HTTP {error.code}）"


def _open_request(request, timeout):
    """发送模型服务请求并返回响应正文。"""
    try:
        with _urllib_request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except _urllib_error.HTTPError as exc:
        raise RuntimeError(_read_http_error(exc)) from exc
    except _urllib_error.URLError as exc:
        reason = str(exc.reason) if exc.reason else str(exc)
        raise RuntimeError(f"无法连接模型服务：{reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"模型服务请求超时（{timeout} 秒）") from exc


def _send_json_request(request, timeout):
    """发送模型服务请求，并将响应解析为 JSON 数据。"""
    raw = _open_request(request, timeout)
    try:
        return _json.loads(raw)
    except (_json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("模型服务返回的内容不是有效 JSON") from exc


def _call_openai_models(base_url, api_key, timeout=_REQUEST_TIMEOUT):
    """调用 OpenAI 兼容的 Models 接口并返回模型名称列表。"""
    request = _urllib_request.Request(
        _build_models_url(base_url),
        headers=_build_headers(api_key),
        method="GET",
    )
    data = _send_json_request(request, timeout)
    return _extract_model_names(data)


def _call_openai_chat(
        base_url, model_name, api_key, system_prompt, prompt,
        timeout=_REQUEST_TIMEOUT):
    """调用 OpenAI 兼容的 Chat Completions 接口并返回助手文本。"""
    selected_model = str(model_name or "").strip()
    if not selected_model or selected_model.lower() == _NONE_MODEL.lower():
        raise RuntimeError("模型名称不能为空")

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": str(system_prompt or "")},
            {"role": "user", "content": str(prompt or "")},
        ],
        "max_tokens": _MAX_LENGTH,
        "stream": False,
    }
    request = _urllib_request.Request(
        _build_chat_completions_url(base_url),
        data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_build_headers(api_key),
        method="POST",
    )
    data = _send_json_request(request, timeout)
    return _extract_response_text(data)


def _call_model_unload(
        base_url, model_name, api_key, timeout=_REQUEST_TIMEOUT):
    """调用 llama.cpp 路由器接口卸载指定模型。"""
    payload = {"model": str(model_name).strip()}
    request = _urllib_request.Request(
        _build_unload_url(base_url),
        data=_json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_build_headers(api_key),
        method="POST",
    )
    _open_request(request, timeout)


def _try_model_unload(base_url, model_name, api_key):
    """尝试卸载模型；失败时发出警告并保留已生成结果。"""
    try:
        _call_model_unload(base_url, model_name, api_key)
    except Exception as exc:
        _warnings.warn(
            f"H3PromptEnhance 自动卸载模型失败：{exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _build_node_output(prompt, model_names):
    """创建包含节点输出与前端动态模型列表的 ComfyUI 返回结构。"""
    models_text = chr(10).join(model_names)
    return {
        "ui": {"models": list(model_names)},
        "result": (prompt, models_text),
    }


class H3PromptEnhance:
    """使用本地 OpenAI 兼容 LLM 服务增强用户提示词。"""

    @classmethod
    def INPUT_TYPES(cls):
        """声明按使用流程排列的模型服务参数与提示词端点。"""
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8080/v1",
                    "tooltip": "本地 OpenAI 兼容模型服务的基础地址。",
                }),
                "api_key": ("STRING", {
                    "default": "1234",
                    "tooltip": "模型服务 API Key；服务无需认证时可留空。",
                }),
                "system_skill": (_get_skill_names(), {
                    "default": _CUSTOM_SKILL,
                    "tooltip": "选择 skills 目录中的 Skill；Custom 使用下方自定义系统提示词。",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom 模式下发送给大模型的系统提示词。",
                }),
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "连接需要增强的提示词文本。",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "改变此值可触发节点重新执行。",
                }),
                "model_name": ("STRING", {
                    "default": _NONE_MODEL,
                    "tooltip": "首次选择 None 执行以获取模型列表，随后从下拉列表选择模型。",
                }),
                "model_auto_unload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "生成完成后通过 llama.cpp 管理接口卸载当前模型。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "models")
    FUNCTION = "enhance_prompt"
    CATEGORY = "H3/text"
    DESCRIPTION = "使用本地 OpenAI 兼容 LLM 服务增强提示词并返回可用模型列表。"

    def enhance_prompt(
            self, base_url, api_key, system_prompt, prompt, model_name,
            model_auto_unload, system_skill=_CUSTOM_SKILL, seed=0):
        """获取模型列表，选择系统 Skill，增强提示词并按配置卸载模型。"""
        del seed  # seed 仅用于 ComfyUI 触发重新执行，不参与模型请求。
        model_names = _call_openai_models(
            base_url=base_url,
            api_key=api_key,
        )
        selected_model = str(model_name or "").strip()
        if (not selected_model
                or selected_model.lower() == _NONE_MODEL.lower()
                or selected_model not in model_names):
            return _build_node_output(str(prompt or ""), model_names)

        final_system_prompt = _read_system_skill(system_skill, system_prompt)
        enhanced_prompt = _call_openai_chat(
            base_url=base_url,
            model_name=selected_model,
            api_key=api_key,
            system_prompt=final_system_prompt,
            prompt=prompt,
        )
        if model_auto_unload:
            _try_model_unload(base_url, selected_model, api_key)
        return _build_node_output(enhanced_prompt, model_names)


NODE_CLASS_MAPPINGS = {
    "H3PromptEnhance": H3PromptEnhance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PromptEnhance": "H3 Prompt Enhance ( OpenAI )",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
