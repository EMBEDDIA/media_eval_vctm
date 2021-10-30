import argparse
import numpy as np
import os
import pickle

from utils import load_data

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluation.measures import CoherenceCV, InvertedRBO, TopicDiversity

parser = argparse.ArgumentParser()
parser.add_argument("--text_model", type=str, required=True)
parser.add_argument("--tries", type=int, default=5, required=False)
args = parser.parse_args()

model_type, n_topics, _ = os.path.basename(args.text_model).split("_")
image_training_dataset = pickle.load(open(os.path.join(args.text_model, "image_training_dataset.pkl"), "rb"))

image_training_dataset.X_contextual = np.array(image_training_dataset.X_contextual.cpu())

preprocessed_split = pickle.load(open(os.path.join(args.text_model, "preprocessed_split.pkl"), "rb"))


for t in range(args.tries):
    model_dir = "models/image_student/%s_%s_%s/" %(model_type, n_topics, t)
    os.makedirs(model_dir)
    log = open(os.path.join(model_dir,"log.txt"), "w")
    print("Text model: %s" %os.path.abspath(args.text_model), file=log)
    
    vctm = ZeroShotTM(bow_size=len(image_training_dataset.idx2token), contextual_size=512, n_components=int(n_topics), model_type=model_type)
    vctm.fit(image_training_dataset, save_dir=model_dir, verbose=True)
    
    topics = vctm.get_topic_lists(25)
    irbo = InvertedRBO(topics=topics)
    td = TopicDiversity(topics=topics)   
    cv = CoherenceCV(texts=preprocessed_split, topics=topics)

    print("IRBO: %2.4f" %irbo.score(), file=log)
    print("Diversity: %2.4f" %td.score(), file=log)
    print("Coherence: %2.4f" %cv.score(), file=log)
    log.close()

