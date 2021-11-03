import argparse
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import scipy

import warnings
warnings.filterwarnings('ignore')


from collections import defaultdict
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import Matches, KLDivergence
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import load_data, load_model


parser = argparse.ArgumentParser()
parser.add_argument("--text_model", type=str, required=True)
parser.add_argument("--image_model", type=str, required=True)
parser.add_argument("--test", type=str, default="test.pkl", required=False)
args = parser.parse_args()


test_texts, test_imgs = load_data(args.test)
tp = pickle.load(open(os.path.join(args.text_model, "tp.pkl"), "rb"))
ctm = load_model(args.text_model, len(tp.vocab))
vctm = load_model(args.image_model, len(tp.vocab))

ctm_mat = ctm.get_topic_word_distribution()
vctm_mat = vctm.get_topic_word_distribution()

print("Computing similarities")
sims = defaultdict(list)
for i in range(ctm_mat.shape[0]):
    for j in range(vctm_mat.shape[0]): 
        sim = (cosine_similarity(ctm_mat[i].reshape(1, -1), vctm_mat[j].reshape(1, -1)))
        sims[sim[0][0]].append((i,j)) 


overlap = {}
text_topics = ctm.get_topic_lists(20)
image_topics = vctm.get_topic_lists(20)
for i in range(len(text_topics)):
    for j in range(len(image_topics)):
        o = list(set(text_topics[i]) & set(image_topics[j]))
        overlap[(i,j)] = len(o)

v2t = {}

v_linked = [] 
t_linked = []
for s in sorted(sims, reverse=True):
    for t in sorted(sims[s], key=lambda t: overlap[(t[0],t[1])], reverse=True):
        if not t[0] in t_linked:
            if not t[1] in v_linked:
                v2t[t[1]] = t[0]
                t_linked.append(t[0])
                v_linked.append(t[1])


testing_dataset = tp.transform(text_for_contextual=test_texts)

img_model = SentenceTransformer('clip-ViT-B-32')
test_img_emb = img_model.encode(test_imgs, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
test_img_emb = np.array(test_img_emb.cpu())


image_test_bow_embeddings = scipy.sparse.csr_matrix(np.zeros((len(test_img_emb), 1)))
image_testing_dataset = CTMDataset(X_contextual = test_img_emb, X_bow=image_test_bow_embeddings ,idx2token = testing_dataset.idx2token)

test_topic_dist = ctm.get_doc_topic_distribution(testing_dataset, n_samples=20)
v_test_topic_dist = vctm.get_doc_topic_distribution(image_testing_dataset, n_samples=20)

mapped_v_test_topic_dist = np.zeros_like(v_test_topic_dist)
for v, t in v2t.items():
    mapped_v_test_topic_dist[:,t] = v_test_topic_dist[:,v]

dist_sim = cosine_similarity(test_topic_dist, mapped_v_test_topic_dist)

out = open(os.path.join(args.image_model, "eval.txt"), "w")

mrr = 0
r_1 = 0
r_5 = 0
r_10 = 0
r_50 = 0
r_100 = 0
for doc in tqdm(range(dist_sim.shape[0])):
    ind_sims = sorted([(s,i) for i, s in enumerate(dist_sim[doc])], reverse=True)
    ind_sims = [i[1] for i in ind_sims]
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

mrr = mrr / dist_sim.shape[0]
print("MRR\tR1\tR5\tR10\tR50\tR100")
print("%2.4f\t%d\t%d\t%2d\t%d\t%d" %(mrr, r_1, r_5, r_10, r_50, r_100))

print("MRR\tR1\tR5\tR10\tR50\tR100", file=out)
print("%2.4f\t%d\t%d\t%2d\t%d\t%d" %(mrr, r_1, r_5, r_10, r_50, r_100), file=out)


r_1 = r_1 / dist_sim.shape[0]
r_5 = r_5 / dist_sim.shape[0]
r_10 = r_10 / dist_sim.shape[0]
r_50 = r_50 / dist_sim.shape[0]
r_100 = r_100 / dist_sim.shape[0]


print("MRR\tR1\tR5\tR10\tR50\tR100")
print("%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f" %(mrr, r_1, r_5, r_10, r_50, r_100))

print("MRR\tR1\tR5\tR10\tR50\tR100", file=out)
print("%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f\t%2.4f" %(mrr, r_1, r_5, r_10, r_50, r_100), file=out)


mt = Matches(test_topic_dist, mapped_v_test_topic_dist)
print("Matches: %2.2f", mt.score())
print("Matches: %2.2f", mt.score(), file=out)

kl = KLDivergence(test_topic_dist, mapped_v_test_topic_dist)
print("KLD: %2.2f" %kl.score())
print("KLD: %2.2f" %kl.score(), file=out)

print(v2t, file=out)
out.close()
