import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import os

from sentence_transformers import SentenceTransformer, util

text_list = pd.read_csv(open("./Data2021/MediaEvalNewsImagesBatch04articles.tsv", "r"), sep="\t")
image_list = pd.read_csv(open("./Data2021/MediaEvalNewsImagesBatch04images.tsv", "r"), sep="\t")

texts = []
text_ids = []
for _, r in tqdm(text_list.iterrows(), desc="Loading texts"):
    if not pd.isnull(r.text):
        texts.append(r.title + ". " + r.text) 
        text_ids.append(r.articleID)

images = []
image_ids = []

for _, r in tqdm(image_list.iterrows(), desc="loading images"):
    if not pd.isnull(r.imgFile) and os.path.exists(os.path.join("./Data2021/images", r.imgFile)):
        img = Image.open(os.path.join("./Data2021/images", r.imgFile))
        img = img.convert("RGB")
        images.append(img)
        image_ids.append(r.imgFile)



model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
img_model = SentenceTransformer('clip-ViT-B-32')

img_emb = img_model.encode(images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
img_emb = np.array(img_emb.cpu())

with open("b_submission.csv", "w") as out:
    for doc in tqdm(range(len(texts)), desc="Searching images"):
        query_emb = model.encode([texts[doc]], convert_to_tensor=True, show_progress_bar=False)
        hits = util.semantic_search(query_emb, img_emb, top_k=100)[0]
        ind_sims = [hit['corpus_id'] for hit in hits]
        img_ids = [image_ids[i] for i in ind_sims] 
        print(str(int(text_ids[doc]))+"\t"+"\t".join(img_ids), file=out)
