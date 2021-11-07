import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
from contextualized_topic_models.models.ctm import ZeroShotTM
import numpy as np

def load_data(filename):
    print("Loading data from %s" %filename)
    test = pd.read_pickle(open(filename, 'rb'))

    test_texts = []
    test_imgs = []

    for _, r in tqdm(test.iterrows(), desc="Finding texts and images"):
        if os.path.exists(os.path.join("./Data2021/images", r.imgFile)):
            if not pd.isnull(r.text):
                test_texts.append(r.title + ". " + r.text) 
                img = Image.open(os.path.join("./Data2021/images", r.imgFile))
                img = img.convert("RGB")
                test_imgs.append(img)

    return test_texts, test_imgs


def load_model(model_folder, bow_size, contextual_size=512, epoch=99):
    model_type, n_topics = os.path.basename(model_folder).split("_")[:2]
    for d in os.listdir(model_folder):
        if d.startswith("contextualized_topic_model"):
            ctm = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size, n_components=int(n_topics))
            model_path = os.path.join(model_folder, d)
            print("Loading model from %s" %model_path)
            ctm.load(model_path, epoch=epoch)
            return ctm


def sharpen_topic_distirbution(doc_topics):
    num_topics = doc_topics.shape[1]
    doc_topics[doc_topics < 1/num_topics] = 0.000000000001
    doc_topics = doc_topics/doc_topics.sum(axis=1)[:, np.newaxis]
    return doc_topics
