import sys
from comfy_api.latest import ComfyExtension, io

# 用于计算 H3 模型的视频时长：秒（seconds） --> 长度（length）的换算
# length = max(5, round(seconds * 24)) + (5 - (max(5, round(seconds * 24)) % 17)) % 17

class H3DurationLength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3DurationLength",
            display_name="H3 Duration Length",
            category="utilities/primitive",
            inputs=[
                io.Float.Input("seconds", min=0.1, max=sys.maxsize, step=0.1,
                               tooltip="Convert the time seconds into the video length of h3"),
            ],
            outputs=[io.Int.Output(display_name="length")],
        )

    @classmethod
    def execute(cls, seconds: float) -> io.NodeOutput:
        length = max(5, round(seconds * 24)) + (5 - (max(5, round(seconds * 24)) % 17)) % 17
        return io.NodeOutput(length)
    
    
NODE_CLASS_MAPPINGS = {
    "H3DurationLength": H3DurationLength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DurationLength": "H3 Duration Length",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}