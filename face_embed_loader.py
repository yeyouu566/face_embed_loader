
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

    def load_embedding(self, pt_path):
        vec = torch.load(pt_path)
        if len(vec.shape) == 1:
            vec = vec.unsqueeze(0)
        return (vec,)
