"""
Minimax H3 Latent Upscaler - ComfyUI 推理节点 (纯3D卷积版本)
- 自动检测模型结构 (通道数、块数、Temporal 位置)
- 支持 FP32 / FP16 / BF16 推理
- 强制关闭注意力 (attn=False)
- 模型从 ComfyUI/models/latent_upscale_models/ 加载
- 推理结束后将模型踢回 CPU，下次从缓存取出时再搬回设备
- https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import folder_paths
import re
import math
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent
from einops import rearrange

# ==========================================
# 注册模型文件夹
# ==========================================
_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

# ==========================================
# Minimax H3 归一化参数 (来自训练代码，24通道)
# ==========================================
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, 
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, 
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, 
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, 
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, 
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD  = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

def _make_norm_tensors(device, dtype):
    """创建与 3D latent 张量形状兼容的归一化参数。"""
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


# MiniMax H3 VAE 空间下采样倍数（像素边长 / latent 网格）。
_LATENT_SPATIAL_FACTOR = 16


def _align_resolution_to_32(width, height):
    """校验像素宽高为正数，并分别向上对齐到 32 的倍数。"""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("宽度和高度必须都大于 0")
    return ((width + 31) // 32) * 32, ((height + 31) // 32) * 32


def _calculate_resolution_from_long_side(source_width, source_height, long_side):
    """按源比例计算目标像素尺寸，并将长短边分别向上对齐到 32 的倍数。"""
    long_side = int(long_side)
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

# ==========================================
# 3D 网络组件 (与训练代码一致)
# ==========================================
def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = rearrange(self.q(h), "b c t h w -> b 1 (t h w) c")
        k = rearrange(self.k(h), "b c t h w -> b 1 (t h w) c")
        v = rearrange(self.v(h), "b c t h w -> b 1 (t h w) c")
        h = F.scaled_dot_product_attention(q, k, v)
        h = rearrange(h, "b 1 (t h w) c -> b c t h w", t=x.shape[2], h=x.shape[3], w=x.shape[4])
        return x + self.proj_out(h)

class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h

class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h

# ==========================================
# 纯3D主干网络 (与训练代码一致)
# ==========================================
class LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        
        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(AttnBlock3D(channels))
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))
                
        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(AttnBlock3D(channels))
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))
                
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            # 计算目标大小 (T, H, W)
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        # 三线性插值
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

# ==========================================
# 模型加载 (纯3D版本)
# ==========================================
MODEL_CACHE = {}

def get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]

def scan_models():
    files = []
    model_dir = get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
    return names if names else [f"(请将模型放入: {model_dir})"]

def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # 移除可能的前缀 (如果有) 并处理 FP8 格式
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd

def _extract_upscaler_sd(sd):
    # 兼容之前可能包含 'upscaler.' 前缀的合并权重
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd

def _detect_arch(sd):
    """
    从 state_dict 推断模型结构参数。
    返回: dict 包含 in_blocks, out_blocks, channels, in_channels, dropout, attn, temporal_every, temporal_kernel
    """
    # 默认参数与 Minimax H3 训练配置保持一致
    cfg = {
        "in_channels": 24,
        "in_blocks": 12,
        "out_blocks": 12,
        "channels": 512,
        "dropout": 0.1,
        "attn": False,
        "temporal_every": 2,   
        "temporal_kernel": 5,
    }

    # 检测通道数
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    # 检测 in_blocks 和 out_blocks 数量
    in_ids = set()
    out_ids = set()
    temporal_in_indices = set()
    temporal_out_indices = set()
    for k in sd.keys():
        # 匹配 in_blocks 中的 ResBlock (有 in_layers)
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        # 匹配 out_blocks 中的 ResBlock
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        # 匹配 temporal 层 (dwconv)
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    # 检测 temporal 配置
    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2  # 训练默认
        # 尝试从 key 中读取 kernel 大小
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                # 读取形状获取 kernel size (维度是 (out, in/groups, T, H, W))
                kernel_t = sd[k].shape[2]
                cfg["temporal_kernel"] = kernel_t
                break
    else:
        cfg["temporal_every"] = 0  # 无 temporal

    # 检测 attn (如果存在 attn 层键)
    if any('attn' in k for k in sd):
        cfg["attn"] = True 

    # 推理时为了性能和稳定性，强制 attn=False
    cfg["attn"] = False

    return cfg

def load_model(name, device, precision):
    """加载或从缓存取出放大模型，并确保权重位于目标设备。"""
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in MODEL_CACHE:
        # 上次推理可能已把模型踢回 CPU，这里再搬回目标设备。
        return MODEL_CACHE[cache_key].to(device)

    path = os.path.join(get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)

    cfg = _detect_arch(up_sd)

    # 构建模型
    model = LatentResizer3D(
        in_channels=cfg["in_channels"],
        in_blocks=cfg["in_blocks"],
        out_blocks=cfg["out_blocks"],
        channels=cfg["channels"],
        dropout=cfg["dropout"],
        attn=cfg["attn"],           # 强制 False
        temporal_every=cfg["temporal_every"],
        temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)  # 严格匹配，确保结构一致
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval()
    if dtype != torch.float32:
        model = model.to(dtype)

    MODEL_CACHE[cache_key] = model

    print(f"[MinimaxH3-3D] 加载放大模型: {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Attn: 强制关闭 | Temporal: {'✓' if cfg['temporal_every']>0 else '✗'} "
          f"(every={cfg['temporal_every']}, kernel={cfg['temporal_kernel']})")
    return model

# ==========================================
# ComfyUI 节点
# ==========================================
def _unpack_av_node_output(result):
    """解包 ComfyUI 核心 AV 节点的返回值，兼容 NodeOutput 与裸元组。"""
    if hasattr(result, "args"):
        return result.args
    return result


class _LazyUpscaleModel:
    """按需加载专用 3D 放大模型，避免同尺寸短路时无谓读盘。"""

    def __init__(self, model_name, device, precision):
        """记录加载参数，延迟到首次调用时再真正加载。"""
        self.model_name = model_name
        self.device_name = device
        self.precision = precision
        self.model = None
        self.device = None

    def __call__(self):
        """返回 (model, device)；首次调用时加载并缓存。"""
        if self.model is None:
            self.device = torch.device(
                self.device_name if torch.cuda.is_available() else "cpu")
            self.model = load_model(self.model_name, self.device, self.precision)
        return self.model, self.device

    def offload_to_cpu(self):
        """推理结束后把模型踢回 CPU，释放显存给后续节点。"""
        if self.model is None:
            return
        # 同一次 run 内 samples/positive/negative 共用该模型，全部完成后再卸载。
        self.model.to("cpu")
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("[MinimaxH3-3D] 模型已卸载到 CPU，释放显存给后续节点")


def _spatial_hw_from_video(video):
    """从 video latent 读取空间网格高宽。

    5D 为 [B,C,T,H,W]；4D 末两维始终是 H/W（采样为 [B,C,H,W]，条件为 [C,T,H,W]）。
    """
    if video is None:
        raise ValueError("video tensor is missing")
    ndim = getattr(video, "ndim", 0)
    if ndim == 5:
        return int(video.shape[3]), int(video.shape[4])
    if ndim == 4:
        return int(video.shape[2]), int(video.shape[3])
    raise ValueError(
        "expected video [B,C,T,H,W] or 4D, got shape %s"
        % (tuple(getattr(video, "shape", ())),))


def _upscale_video_tensor_with_model(
        video, target_h, target_w, get_model, precision, layout="cthw"):
    """使用专用 3D 放大模型缩放 video latent 的空间维。

    - 仅改 H/W；T、C、batch 不变。
    - 已是目标网格则原样返回。
    - layout='cthw' 时 4D 视为 [C,T,H,W]（conditioning 中的 keyframe/ref）。
    - layout='bcthw' 时 4D 视为 [B,C,H,W]（采样 latent）。
    """
    if video is None:
        raise ValueError("video tensor is missing")
    if getattr(video, "ndim", 0) not in (4, 5):
        raise ValueError(
            "expected video latent [B,C,T,H,W] or 4D, got shape %s"
            % (tuple(getattr(video, "shape", ())),))

    src_h, src_w = _spatial_hw_from_video(video)
    if src_h == target_h and src_w == target_w:
        return video

    if video.ndim == 5:
        channels = int(video.shape[1])
    elif layout == "bcthw":
        channels = int(video.shape[1])
    else:
        channels = int(video.shape[0])
    expected_c = len(LATENTS_MEAN)
    if channels != expected_c:
        raise ValueError(
            "放大模型仅支持 %d 通道 H3 video latent，实际为 %d 通道"
            % (expected_c, channels))

    model, dev = get_model()
    compute_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    orig_dtype = video.dtype
    orig_ndim = video.ndim
    s = video.clone()
    # 统一成 5D (B, C, T, H, W) 再送入 Conv3d。
    if orig_ndim == 4:
        if layout == "bcthw":
            s = s.unsqueeze(2)
        else:
            s = s.unsqueeze(0)

    s = s.to(dev, compute_dtype)
    norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)
    s = (s - norm_mean) / norm_std

    # 按当前张量自身的源网格计算 scale，并显式锁定时间维。
    T = int(s.shape[2])
    target_scale = ((target_w / src_w) * (target_h / src_h)) ** 0.5
    target_size = (T, target_h, target_w)
    with torch.no_grad():
        out = model(s, scale=target_scale, target_size=target_size)

    out = out * norm_std + norm_mean
    if orig_ndim == 4:
        if layout == "bcthw":
            out = out.squeeze(2)
        else:
            out = out.squeeze(0)
    return out.cpu().to(orig_dtype)


def _upscale_keyframe_entry(kf, target_h, target_w, get_model, precision):
    """使用 3D 放大模型缩放单条 minimax_keyframes 中的 video latent。"""
    if not isinstance(kf, dict):
        return kf
    out = dict(kf)
    if "latent" in out and out["latent"] is not None:
        out["latent"] = _upscale_video_tensor_with_model(
            out["latent"], target_h, target_w, get_model, precision, layout="cthw")
    return out


def _upscale_ref_entry(ref, target_h, target_w, get_model, precision):
    """使用 3D 放大模型缩放 minimax_refs 中带视频 latent 的引用块。

    - image / video / video_audio：用模型放大 latent，并同步 latent_h / latent_w。
    - audio 纯音频块：原样浅拷贝。
    - audio_latent / ref_audio_t 永不放大。
    """
    if not isinstance(ref, dict):
        return ref
    out = dict(ref)
    kind = out.get("kind", "")
    has_video = "latent" in out and out["latent"] is not None
    if kind == "audio" or not has_video:
        return out
    scaled = _upscale_video_tensor_with_model(
        out["latent"], target_h, target_w, get_model, precision, layout="cthw")
    out["latent"] = scaled
    # 同步 stock ref2va 使用的网格元数据
    if getattr(scaled, "ndim", 0) == 5:
        out["latent_h"] = int(scaled.shape[3])
        out["latent_w"] = int(scaled.shape[4])
        if "latent_t" in out:
            out["latent_t"] = int(scaled.shape[2])
    elif getattr(scaled, "ndim", 0) == 4:
        out["latent_h"] = int(scaled.shape[2])
        out["latent_w"] = int(scaled.shape[3])
    return out


def _upscale_conditioning(conditioning, target_h, target_w, get_model, precision):
    """使用 3D 放大模型同步缩放 CONDITIONING 内的视频 keyframe / 视频类 refs。

    - minimax_keyframes[*].latent -> 目标网格
    - minimax_refs 中带 video latent 的 image/video 块同上；audio 不动
    - 文本 embedding、minimax_frame_count、时间锚点不改
    - 返回新列表；输入为 None 则返回 None
    """
    if conditioning is None:
        return None
    out = []
    for entry in conditioning:
        if not entry or len(entry) < 2:
            out.append(entry)
            continue
        emb, extra = entry[0], entry[1]
        d = extra.copy() if isinstance(extra, dict) else {}

        prior = d.get("minimax_keyframes")
        if prior:
            d["minimax_keyframes"] = [
                _upscale_keyframe_entry(
                    kf, target_h, target_w, get_model, precision)
                for kf in prior
            ]

        refs = d.get("minimax_refs")
        if refs:
            d["minimax_refs"] = [
                _upscale_ref_entry(
                    ref, target_h, target_w, get_model, precision)
                for ref in refs
            ]

        out.append([emb, d])
    return out


class H3LatentUpscalerNode3DV3:
    """Minimax H3 Latent 放大节点，按长边分辨率放大视频并同步缩放条件。"""
    @classmethod
    def INPUT_TYPES(cls):
        """声明节点输入：目标长边、推理参数，以及可选的正负条件。"""
        return {
            "required": {
                "latent": ("LATENT",),
                "model_name": (scan_models(),),
                "long_side": ("INT", {
                    "default": 768, "min": 1, "step": 1,
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "run"
    CATEGORY = "video/MinimaxH3"

    def run(self, latent, model_name, device, precision,
            long_side=768, positive=None, negative=None):
        """按目标长边放大视频 latent，并用同一模型同步放大 positive/negative。"""
        if model_name.startswith('('):
            raise ValueError("请将模型文件放入 latent_upscale_models 目录")

        # 使用 ComfyUI 核心节点分离 AV latent，避免音频参与视频放大。
        video_latent, audio_latent = _unpack_av_node_output(
            LTXVSeparateAVLatent.execute(latent))
        source_samples = video_latent["samples"]
        source_height = int(source_samples.shape[-2])
        source_width = int(source_samples.shape[-1])

        # H3 VAE 空间压缩比为 16，不能按 32 换算，否则 768 源会被当成 1536。
        target_width_pixels, target_height_pixels = _calculate_resolution_from_long_side(
            source_width * _LATENT_SPATIAL_FACTOR,
            source_height * _LATENT_SPATIAL_FACTOR,
            long_side)
        target_width = target_width_pixels // _LATENT_SPATIAL_FACTOR
        target_height = target_height_pixels // _LATENT_SPATIAL_FACTOR
        # 目标小于当前时保持原网格，兼容误设更小长边，避免把放大节点当成缩小。
        if target_width < source_width or target_height < source_height:
            target_width = source_width
            target_height = source_height

        # 采样与条件共用同一放大模型；仅在确实需要放大时才加载。
        get_model = _LazyUpscaleModel(model_name, device, precision)
        sample_layout = "bcthw" if source_samples.ndim == 4 else "cthw"
        video_latent["samples"] = _upscale_video_tensor_with_model(
            source_samples, target_height, target_width,
            get_model, precision, layout=sample_layout)
        out_latent, = _unpack_av_node_output(
            LTXVConcatAVLatent.execute(video_latent, audio_latent))
        # 同步放大 keyframe / 视频类 refs，避免第二阶段 PackedLayout 行数不一致。
        out_pos = _upscale_conditioning(
            positive, target_height, target_width, get_model, precision)
        out_neg = _upscale_conditioning(
            negative, target_height, target_width, get_model, precision)

        # samples 与条件全部推理完成后再卸载，避免 keyframe/ref 放大时反复搬模型。
        get_model.offload_to_cpu()
        return (out_latent, out_pos, out_neg)

# ==========================================
# 节点注册
# ==========================================
NODE_CLASS_MAPPINGS = {
    "H3LatentUpscalerNode3DV3": H3LatentUpscalerNode3DV3,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LatentUpscalerNode3DV3": "H3 Latent Upscaler V3 (3D)",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}