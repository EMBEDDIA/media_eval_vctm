import argparse
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, default="test.pkl", required=False)
args = parser.parse_args()

test_texts, test_imgs = load_data(args.test)

model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
img_model = SentenceTransformer('clip-ViT-B-32')

img_emb = img_model.encode(test_imgs, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
img_emb = np.array(img_emb.cpu())

mrr = 0
r_1 = 0
r_5 = 0
r_10 = 0
r_50 = 0
r_100 = 0

for doc in tqdm(range(len(test_texts)), desc="Searching images"):
    query_emb = model.encode([test_texts[doc]], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, img_emb, top_k=100)[0]
    ind_sims = [hit['corpus_id'] for hit in hits]
    if ind_sims[0] == doc:
        r_1 += 1
    if doc in ind_sims[:5]:
        r_5 += 1
    if doc in ind_sims[:10]:
        r_10 += 1
    if doc in ind_sims[:50]:
        r_50 += 1
    if doc in ind_sims[:100]:
        r_100 += 1
        mrr += 1.0 / (ind_sims[:100].index(doc)+1)

mrr = mrr / len(test_texts)
print("MRR\tR1\tR5\tR10\tR50\tR100")
print("%2.4f\t%d\t%d\t%2d\t%d\t%d" %(mrr, r_1, r_5, r_10, r_50, r_100))

r_1 = r_1 / len(test_texts)
r_5 = r_5 / len(test_texts)
r_10 = r_10 / len(test_texts)
r_50 = r_50 / len(test_texts)
r_100 = r_100 / len(test_texts)

print("MRR\tR1\tR5\tR10\tR50\tR100")
print("%2.4f\t%2.4f\t%2.4f\t%2d\t%2.4f\t%2.4f" %(mrr, r_1, r_5, r_10, r_50, r_100))

