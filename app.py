import base64
from io import BytesIO
from potassium import Potassium, Request, Response

from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
from PIL import Image
import numpy as np
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    
    context = {
        "model": model,
        "processor": processor,
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    url = request.json.get("image")
    processor = context.get("processor")
    model = context.get("model")
    # read base64 image
    image = Image.open(BytesIO(base64.b64decode(url)))
    # image = Image.open(requests.get(url, stream=True).raw)
    # # prepare image for the model
    inputs = processor(image, return_tensors="pt")
    # # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)

    # # save output
    output_image = Image.fromarray(output)
    # png 
    output_image_base64 = base64.b64encode(output_image.tobytes()).decode("utf-8")
    data_url = f"data:image/png;base64,{output_image_base64}"
    output_image.save("output.png")


    return Response(
        json = {"outputs": data_url},
        status=200
    )

if __name__ == "__main__":
    app.serve()