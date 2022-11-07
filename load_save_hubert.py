"""
Compute the Self Supervised Learning (SSL) features (HuBERT or wav2vec2) for a given model and store them
on local disk.

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk
python load_save.py HUBERT_BASE ./data/train.csv ./data/hubert_features/train/ ./data/LibriSpeech/

"""

import argparse
import os
import sys
from feature_extractor import load_model,SSL_features,clip_features
import torch
from tqdm import tqdm
import pandas as pd
from os import path
import glob


possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H"]


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#

def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0],add_help=True)
    parser.add_argument("model",type=str,help = "name of the model for example, HUBERT_BASE",choices = possible_models)
    parser.add_argument("metadata_file_path", type = str,help = "a text file or dataframe containing paths of wave files, words, start point, duration")
    parser.add_argument('-metadata_file_list', nargs="+", default=["train.csv", "val.csv","test.csv"], help = " list of metadata files")
    parser.add_argument("path_to_output", type = str, help = "path to output folder where features will be stored")
    parser.add_argument("path_to_libri", type = str, help = "base path to librispeech dataset")
    parser.add_argument("layer", type = int, help = "layer you want to extract",nargs='?',default=12)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def split_string(strs):
    return strs.split(sep="/")[-2]

#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#

def main():
    args = check_argv()

    # Check whether the specified metadata file exists or not
    isExist = os.path.exists(args.metadata_file_path)

    if not isExist:
        print(args.metadata_file_path)
        print("provide the correct path for the text/dataframe file having list of wave files")
        sys.exit(1)

    # Check whether the specified libripseech dataset path exists or not
    isExist = os.path.exists(args.path_to_libri)

    if not isExist:
        print("provide the correct path for the librispeech dataset")
        sys.exit(1)

    # Check whether the specified output path exists or not
    for m in args.metadata_file_list:

        isExist = os.path.exists(path.join(args.path_to_output,m.split(".")[0]))
    
        # Create a new directory for output because it does not exist 
        if not isExist:
            os.makedirs(path.join(args.path_to_output,m.split(".")[0]))
            print("The new directory for output is created!",m.split(".")[0])

    # Load model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,sr = load_model(args.model,device)

    ## Extraction of SSL features ( HuBERT_BASE in this case )

    for f_name in args.metadata_file_list:
        data = pd.read_csv(os.path.join(args.metadata_file_path,f_name))
        for _,row in tqdm(data.iterrows()):
            file_path = path.join(args.path_to_libri,row["filename_path"].strip("./").strip("\n"))
            word_description = row["word"] + "_" + str(row["start"]) + "_" + str(row["duration"]) + "_" \
            + path.splitext(path.split(row["filename_path"])[-1])[0]
            features = SSL_features(file_path,model,sr,layer=args.layer,device=device)
            word_features = clip_features(features,row["start"],row["duration"],layer=args.layer)
            torch.save(word_features, path.join(args.path_to_output,f_name.split(".")[0],word_description+".pt"))

    
    ## save the metadata file for the extracted features
    
    PATH = args.path_to_output
    my_files = sorted(glob.glob(PATH + '*/**/*.pt',recursive=True))
    print("total calculated features files",len(my_files))
    df_metadata = pd.DataFrame(my_files,columns=["path"])
    df_metadata["partition"] = df_metadata["path"].apply(split_string)
    df_metadata.to_csv(os.path.join(PATH,"hubert_feature_metadata.csv"),index=False)

if __name__ == "__main__":
    main()
