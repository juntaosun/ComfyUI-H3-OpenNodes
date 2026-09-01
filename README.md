# ComfyUI-H3-OpenNodes  
### MiniMax-H3 角色卡 - 支持高低噪放大的视频生成的实用功能节点 !    
### MiniMax-H3 Character Card - A practical feature node supporting high and low noise amplification in video generation.    


<div align="center">
  <img src='./images/image01.jpg' width="75%" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);" />
</div>

--- 
### ✨ 最简单的使用场景 (The simplest use case):  
👉 当你定义了一个角色卡包括图像和声音,比如角色的名称叫:露西.
那么你的提示词只需要写露西在做什么,例如:露西开心的说:"你好",
该节点会自动绑定并关联到该角色卡和声音克隆.

👉 Once you define a character card that includes an image and a voice, such as a character named Lucy, then your prompt only needs to describe what Lucy is doing, for example: Lucy happily says, "Hello." The node will automatically bind and associate with that character card and voice clone.

> 👉 示例工作流 (Example workflow):  ComfyUI-H3-OpenNodes\workflows\

---

🤗 为什么选择它?  Why choose it?    
- ✅集成式媒体加载器: 支持图像,声音,名字,描述.  
- ✅Integrated media loader: Supports images, sounds, names, and descriptions.  
- ✅全自动提示词组装: 自动分析媒体拼装提示词.  
- ✅Fully automatic prompt assembly: Automatically analyzes media and assembles prompts.  
- ✅支持高低噪双采样: 所有条件均支持高低采样.  
- ✅Supports high and low noise dual sampling: High and low sampling are supported under all conditions.  

--- 

🚀 什么是高低双采样? What is high-low dual sampling?    
- **低采样**: 可以使用 512 分辨率进行极速抽卡预览.  
- **Low sampling**: Allows for ultra-fast gacha previews using a 512 resolution.  
- **高采样**: 使用latent高分辨率放大生成清晰的最终视频.  
- **High sampling**: Uses latent high-resolution upsampling to generate a sharp final video.  

---  


> 部份节点功能来自 ComfyUI 官方节点, 部份节点参考了 ComfyUI-PainterNodes 的实现, 在此基础上扩展自己需要的功能.  
Some node functionalities are derived from the official ComfyUI nodes, Some nodes are based on the implementation of ComfyUI-PainterNodes, and additional features are added based on these functionalities.   



