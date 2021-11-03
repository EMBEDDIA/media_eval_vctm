import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

for model in sorted(os.listdir(args.folder)):
    log = os.path.join(args.folder, model, "log.txt")
    if os.path.exists(log):
        lines = open(log,'r').readlines()
        score_dict = {}
        for line in lines:
            score, value = line.split(": ")
            score_dict[score] = value.strip()

        try:
            print(model, score_dict["IRBO"], score_dict["Diversity"], score_dict["Coherence"])
        except:
            print(model)
            continue
