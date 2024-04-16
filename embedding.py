from transformers import AutoImageProcessor, AutoModel

device = "cuda"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)


def embed_picture(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.detach().cpu().numpy()[0].flatten()
    return embedding
