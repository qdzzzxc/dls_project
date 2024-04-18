from transformers import AutoModelForImageClassification, AutoProcessor
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
from scipy import spatial
from PIL import Image
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = 'gatecitypreservation/architectural_styles'
model1_path = 'cvmodel'
model2_path = 'cvmodel_new_3'
model_emb_name = 'clip-ViT-B-32'

processor = AutoProcessor.from_pretrained(model_id)
MODEL1 = AutoModelForImageClassification.from_pretrained(model1_path).to(device)
MODEL2 = AutoModelForImageClassification.from_pretrained(model2_path).to(device)
MODEL_EMB = SentenceTransformer(model_emb_name)

data_emb = pd.read_parquet('embedded.pqt')


# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "vikhyatk/moondream2"
# revision = "2024-04-02"
# MODEL3 = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, revision=revision
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)



def predict_model1(bytes):
    arch_lst = ['Contemporary', 'Victorian', 'Classical', 'Gothic', 'Georgian',
            'Art Deco', 'Renaissance', 'Byzantine', 'Gothic Revival',
            'Romanesque', 'Islamic', 'Medieval', 'Greek Revival',
            'Baroque', 'Tudor', 'Italiante', 'Neo-futurism', 'Rococo']

    arch_lst = sorted(arch_lst)

    ID2LABEL = {i: label for i, label in enumerate(arch_lst)}

    img = Image.open(bytes)

    inputs = processor(img, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = MODEL1(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return ID2LABEL[predicted_label]


def predict_model2(bytes):
    ID2LABEL = {i: arch for i, arch in enumerate(os.listdir('architectural-styles-dataset/'))}
    
    img = Image.open(bytes)

    inputs = processor(img, return_tensors='pt').to(device)

    with torch.no_grad():
        logits = MODEL2(**inputs).logits

    predicted_label = logits.argmax(-1).item()

    return ID2LABEL[predicted_label].replace(' architecture', '')


def predict_model3(bytes, style):
    pass
    # image = Image.open('Cremlin.jpg')
    # enc_image = MODEL3.encode_image(image)
    # return MODEL3.answer_question(enc_image, f"Describe the image (there is {style} architecture)", tokenizer)


def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
    result_distance = spatial.distance.cosine(emb_a, emb_b)
    return result_distance

def get_similar_images(bytes, n):
    img = Image.open(bytes)
    input_vec = MODEL_EMB.encode(img)

    result_df = copy.deepcopy(data_emb)
    result_df['Distance_with_input'] = result_df.apply(lambda x: calculate_cos_dist(input_vec, x['Embedding']), axis=1)
    result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
    result_df_sorted = result_df_sorted[['Style', 'Image', 'Distance_with_input']]
    return result_df_sorted.head(n)