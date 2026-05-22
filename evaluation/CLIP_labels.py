################################################################################
# Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
#
# Author(s):
#  Leonardo Monchieri (leonardo.monchieri@unipd.it)
#  Elena Camuffo (elenacamuffo97@gmail.com)
#  Francesco Barbato (francesco.barbato@dei.unipd.it)
#  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
#  Simone Milani (simone.milani@dei.unipd.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

import torch
import clip
from tqdm import tqdm
import argparse

# -------------------------------
# Paste your label dictionaries
# -------------------------------


DATASETS = {
    #############
    # FUGURINES #
    #############
    "figurines":{
        0 : {
            0: "old camera",
            1: "toy elephant",
            2: "waldo",
            3: "tesla door handle",
            4: "porcelain hand",
            5: "rubber duck with hat",
            6: "rubber duck with buoy",
            7: "pink ice cream",
            8: "red toy chair",
            9: "green apple",
            10: "pikachu",
            11: "red apple",
            12: "spatula",
            13: "jake",
            14: "toy cat statue",
            15: "pirate hat",
            16: "miffy"
        },

        1: {
            0: "rubics cube",
            1: "green apple",
            2: "green toy chair",
            3: "jake",
            4: "old camera",
            5: "pink ice cream",
            6: "pumpkin",
            7: "red apple",
            8: "rubber duck with hat",
            9: "tesla door handle",
            10: "spatula",
            11: "rubber duck with buoy",
            12: "pirate hat"
        },

        2 : {
            0: "rubber duck with hat",
            1: "rubics cube",
            2: "toy elephant",
            3: "green apple",
            4: "jake",
            5: "toy cat statue",
            6: "pikachu",
            7: "porcelain hand",
            8: "red apple",
            9: "waldo",
            10: "pirate hat"
        },

        3: {

            0: "toy elephant",
            1: "pink ice cream",
            2: "porcelain hand",
            3: "green apple",
            4: "green toy chair",
            5: "old camera",
            6: "pink ice cream",
            7: "rubics cube", 
            8: "spatula",
            9: "toy cat statue",
            10: "waldo",
            11: "rubber duck with buoy",
            12: "pirate hat",
            13: "miffy",
            14: "bag",
            15: "rubber duck with hat"

        },
    },
    
    #########
    # RAMEN #
    #########

    "ramen":{
        0:   {
            0: "chopsticks",
            1: "egg",
            2: "nori",
            3: "bowl",
            4: "napkin",
            5: "sake cup",
            6: "wavy noodles",
            7: "kamaboko",
            8: "plate",
            9: "onion segments"
        },

        1 : {
            0: "bowl",
            1: "chopsticks",
            2: "egg",
            3: "nori",
            4: "wavy noodles",
            5: "kamaboko",
            6: "onion segments",
            7: "corn"
        },


        2 : {
            0: "chopsticks",
            1: "egg",
            2: "sake cup",
            3: "napkin",
            4: "wavy noodles",
            5: "kamaboko",
            6: "corn",
            7: "onion segments"
        },

        3 : {
            0: "bowl",
            1: "egg",
            2: "chopsticks",
            3: "sake cup",
            4: "wavy noodles",
            5: "nori",
            6: "napkin",
            7: "kamaboko",
            8: "napkin",
            9: "plate",
            10: "corn",
            11: "onion segments"
        },


        4 : {
            0: "bowl",
            1: "bowl",
            2: "chopsticks",
            3: "sake cup",
            4: "nori",
            5: "egg",
            6: "wavy noodles",
            7: "glass of water",
            8: "sake cup",
            9: "kamaboko",
            10: "spoon",
            11: "napkin",
            12: "napkin",
            13: "plate",
            14: "plate",
            15: "onion segments"
        },


        5 : {
            0: "nori",
            1: "egg",
            2: "sake cup",
            3: "chopsticks",
            4: "wavy noodles",
            5: "kamaboko",
            6: "corn",
            7: "onion segments"
        },


        6 : {
            0: "glass of water",
            1: "nori",
            2: "egg",
            3: "sake cup",
            4: "sake cup",
            5: "bowl",
            6: "bowl",
            7: "chopsticks",
            8: "wavy noodles",
            9: "spoon",
            10: "spoon",
            11: "corn",
            12: "onion segments",
            13: "hand",
            14: "plate",
            15: "kamaboko",
            16: "plate",
            17: "napkin"
        },
    },

    ###########
    # TEATIME #
    ###########
    "teatime" : {
        0: {

            0: "stuffed bear",
            1: "coffee mug",
            2: "bag of cookies",
            3: "sheep",
            4: "apple",
            5: "paper napkin",
            6: "plate",
            7: "tea in a glass",
            8: "bear nose",
            9: "three cookies",
            10: "coffee"
        },


        1: {
            0: "stuffed bear",
            1: "sheep",
            2: "bag of cookies",
            3: "tea in a glass",
            4: "coffee mug",
            5: "plate",
            6: "three cookies",
            7: "hooves",
            8: "paper napkin",
            9: "coffee",
            10: "bear nose"
        },



        2: {
            0: "tea in a glass",
            1: "hooves",
            2: "stuffed bear",
            3: "bag of cookies",
            4: "paper napkin",
            5: "hooves",
            6: "hooves",
            7: "plate",
            8: "apple",
            9: "coffee mug",
            10: "coffee",
            11: "three cookies",
            12: "three cookies"
        },




        3: {
            0: "stuffed bear",
            1: "sheep",
            2: "apple",
            3: "bag of cookies",
            4: "coffee mug",
            5: "tea in a glass",
            6: "bear nose",
            7: "plate",
            8: "three cookies",
            9: "dall-e brand",
            10: "paper napkin"
        },




        4:{
            0: "tea in a glass",
            1: "apple",
            2: "yellow pouf",
            3: "sheep",
            4: "three cookies",
            5: "plate",
            6: "dall-e brand"
        },


        5: {
            0: "tea in a glass",
            1: "paper napkin",
            2: "apple",
            3: "stuffed bear",
            4: "bag of cookies",
            5: "plate",
            6: "coffee mug",
            7: "coffee",
            8: "three cookies"
        },
    },

    #################
    # WALDO KITCHEN #
    #################
    "waldo_kitchen":{
        0 : {
            0: "knife",
            1: "knife",
            2: "knife",
            3: "yellow desk",
            4: "toaster",
            5: "Stainless steel pots",
            6: "pour-over vessel",
            7: "ottolenghi"
        },

        1 : {
            0: "plastic ladle",
            1: "refrigerator",
            2: "pot",
            3: "spatula"
        },

        2 : {
            0: "knife",
            1: "knife",
            2: "knife",
            3: "knife",
            4: "knife",
            5: "ketchup",
            6: "cabinet"
        },


        3 : {
            0: "plate",
            1: "dark cup",
            2: "frog cup",
            3: "spoon",
            4: "spoon",
            5: "sink"
        },

        4 : {
            0: "plate",
            1: "knife",
            2: "red cup",
            3: "sink"
        }
    }
}
# -----------------------------------
# CONFIGURATION
# -----------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"

# -----------------------------------


def compute_clip_text_descriptors(label_list, model_name=MODEL_NAME, device=DEVICE):
    """Compute normalized CLIP text embeddings for each label string."""
    model, _ = clip.load(model_name, device=device)
    model.eval()

    descriptors = {}
    for label in tqdm(label_list, desc="Encoding labels"):
        prompt = f"an image of a {label}"
        tokens = clip.tokenize(prompt).to(device)

        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            #text_feat /= text_feat.norm(dim=-1, keepdim=True)

        descriptors[label] = text_feat.cpu()
    return descriptors


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Open-vocabulary segmentation evaluation")

    parser.add_argument(
        "-dataset",
        type=str, 
        default=None,
        help="Dataset where compute descriptors"
    )
    

    args = parser.parse_args()
     
    SCENE = args.dataset
    OUTPUT_FILE = "./data/"+ SCENE + "/all_text_descriptors.pt"
    
    # 1-Collect all unique labels from all gt_label dicts
    all_labels = set()
    for d in DATASETS[SCENE]:
        all_labels.update(d.values())
    all_labels = sorted(all_labels)

    print(f"Found {len(all_labels)} unique labels")

    # 2-Compute CLIP text descriptors
    label_to_descriptor = compute_clip_text_descriptors(all_labels)

    # 3-Save to file
    torch.save(label_to_descriptor, OUTPUT_FILE)
    print(f"Saved {len(label_to_descriptor)} CLIP text descriptors to {OUTPUT_FILE}")