from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class ImageCaptionTool(BaseTool):
    name = "Image Captioner"
    
    description = "Generate captions for images"

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cuda"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

# Helper function
def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    image = Image.open(image_path).convert('RGB')

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cuda"  # cuda

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption