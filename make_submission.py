import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from contextualized_topic_models.datasets.dataset import CTMDataset
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_model
from sentence_transformers import SentenceTransformer, util
import pickle
import scipy

import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



parser = argparse.ArgumentParser()
parser.add_argument("--text_model", type=str, required=True)
parser.add_argument("--image_model", type=str, required=True)
parser.add_argument("--tp", type=str, required=True)
args = parser.parse_args()

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


tp = pickle.load(open(args.tp, "rb"))
ctm = load_model(args.text_model, len(tp.vocab))
vctm = load_model(args.image_model, len(tp.vocab))


testing_dataset = tp.transform(text_for_contextual=texts)
img_model = SentenceTransformer('clip-ViT-B-32')
img_emb = img_model.encode(images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
img_emb = np.array(img_emb.cpu())

image_test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(img_emb), 1)))
image_testing_dataset = CTMDataset(X_contextual = img_emb, X_bow=image_test_bow_embeddings ,idx2token = testing_dataset.idx2token)

test_topic_dist = ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)
v_test_topic_dist = vctm.get_doc_topic_distribution(image_testing_dataset, n_samples=20)
dist_sim = cosine_similarity(test_topic_dist, v_test_topic_dist)

model_type, n_topics = os.path.basename(args.text_model).split("_")[:2]


with open(model_type+"_"+n_topics+"_submission.csv", "w") as out:
    for doc in tqdm(range(len(texts)), desc="Searching images"):
        ind_sims = sorted([(s,i) for i, s in enumerate(dist_sim[doc])], reverse=True)
        ind_sims = [i[1] for i in ind_sims]
        img_ids = [image_ids[i] for i in ind_sims] 
        print(str(int(text_ids[doc]))+"\t"+"\t".join(img_ids), file=out)


