# 初始化映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 定义注册辅助函数
def _register_nodes_classes(*registries):
    """批量注册节点，忽略为 None 的注册信息"""
    for reg in registries:
        if reg is None: continue
        NODE_CLASS_MAPPINGS.update(reg.get("classes", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(reg.get("names", {}))

# 导入各节点的包装信息 classes
from .H3PromptPeplace import NODE_REGISTRY as _H3PromptPeplace
from .H3AudioMuteStart import NODE_REGISTRY as _H3AudioMuteStart
from .H3_latent_upscaler_3d_v3 import NODE_REGISTRY as _H3_latent_upscaler_3d_v3
from .H3VideoAudioCut import NODE_REGISTRY as _H3VideoAudioCut
from .H3VideoAudioFrom import NODE_REGISTRY as _H3VideoAudioFrom
from .H3AddGuide import NODE_REGISTRY as _H3AddGuide
from .H3ImageToGrid import NODE_REGISTRY as _H3ImageToGrid
from .CNTextOverlay import NODE_REGISTRY as _CNTextOverlay
from .Logic import NODE_REGISTRY as _Logic
from .ImageResolution import NODE_REGISTRY as _ImageResolution
from .H3ReferenceToVideo import NODE_REGISTRY as _H3ReferenceToVideo
from .H3DurationLength import NODE_REGISTRY as _H3DurationLength
from .H3AudioUpload import NODE_REGISTRY as _H3AudioUpload
from .H3MediaLoader import NODE_REGISTRY as _H3MediaLoader
from .H3MediaToVideo import NODE_REGISTRY as _H3MediaToVideo
from .H3MediaPrompt import NODE_REGISTRY as _H3MediaPrompt

# 注册所有各导入的节点 classes
_register_nodes_classes(
    _H3PromptPeplace, 
    _H3AudioMuteStart, 
    _H3_latent_upscaler_3d_v3, 
    _H3VideoAudioCut, 
    _H3VideoAudioFrom,  
    _H3AddGuide,
    _H3ImageToGrid,
    _CNTextOverlay,
    _Logic,
    _ImageResolution,
    _H3ReferenceToVideo,
    _H3DurationLength,
    _H3AudioUpload,
    _H3MediaLoader,
    _H3MediaToVideo,
    _H3MediaPrompt,
)

# 导出静态资源目录以加载前端 JS 插件
__version__ = "1.0.0"
WEB_DIRECTORY = "./web/js"

# 导出最终的节点变量（必须）
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
    "__version__",
    ]
