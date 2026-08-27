class SingleBooleanTrigger:
    '''
    A Boolean Trigger
    
    Inputs:
    bool    - Boolean trigger
    
    Outputs:
    bool    - Boolean value same as input
    '''
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bool": ("BOOLEAN", {
                    "default": False,
                }),                
            },
        }
                
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "SingleBooleanTriggerEx"
    CATEGORY = "H3/Logic"
    
    def SingleBooleanTriggerEx(self, bool):
        return (bool,)
    
    
    
    
NODE_CLASS_MAPPINGS = {
    "SingleBooleanTrigger": SingleBooleanTrigger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SingleBooleanTrigger": "Single Boolean Trigger",
}
NODE_REGISTRY = {
    "classes": NODE_CLASS_MAPPINGS,
    "names": NODE_DISPLAY_NAME_MAPPINGS,
}
