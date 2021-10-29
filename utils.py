import pandas as pd
from tqdm import tqdm
import os
from PIL import Image


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
