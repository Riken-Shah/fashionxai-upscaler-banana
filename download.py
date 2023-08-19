# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import pipeline
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
    # pipeline('fill-mask', model='bert-base-uncased')

if __name__ == "__main__":
    download_model()