from scripts.DScore.models import dist_model as dm
from scripts.DScore.util import util
import numpy as np
from pyprind import prog_bar

def compute_diversity_score(images_A, images_B, use_gpu):
    assert(type(images_A) == list)
    assert(type(images_B) == list)
    assert(len(images_A) == len(images_B))

    img_pair_num = len(images_A)

    ## Initializing the model
    model = dm.DistModel()
    model.initialize(model='net-lin', net='alex', use_gpu=use_gpu)

    distance = []
    for i in prog_bar(range(img_pair_num), title="Calculating Diversity Scores", width=50):
        img_A = util.im2tensor(images_A[i])
        img_B = util.im2tensor(images_B[i])
        dist = model.forward(img_A, img_B)
        distance.append(dist)

    return np.mean(distance), np.std(distance)


if __name__ == '__main__':
    print("please Call from other file.")
