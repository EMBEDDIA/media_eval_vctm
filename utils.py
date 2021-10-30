import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
from contextualized_topic_models.models.ctm import ZeroShotTM

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


def load_model(model_folder, bow_size, epoch=99):
    model_type, n_topics, _ = os.path.basename(model_folder).split("_")
    for d in os.listdir(model_folder):
        if d.startswith("contextualized_topic_model"):
            ctm = ZeroShotTM(bow_size=bow_size, contextual_size=512, num_epochs=100, n_components=int(n_topics))
            model_path = os.path.join(model_folder, d)
            print("Loading model from %s" %model_path)
            ctm.load(model_path, epoch=epoch)
            return ctm
