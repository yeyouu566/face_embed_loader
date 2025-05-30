import torch

class FaceEmbeddingLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pt_path": ("STRING", {"default": "/mnt/data/face_vector.pt"}),
            }
        }

    RETURN_TYPES = ("FACEID_EMBED",)
    RETURN_NAMES = ("faceid_embeds",)
    FUNCTION = "load_embedding"

    CATEGORY = "yeyoung/embedding"

    def load_embedding(self, pt_path):
        vec = torch.load(pt_path)
        if len(vec.shape) == 1:
            vec = vec.unsqueeze(0)
        return (vec,)

# ✅ 반드시 있어야 함!
NODE_CLASS_MAPPINGS = {
    "FaceEmbeddingLoader": FaceEmbeddingLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbeddingLoader": "Face Embedding Loader (by yeyoung)"
}
