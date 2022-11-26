import argparse
import random
import sys
import os
import numpy as np
import torch
from collections import Counter
from os import path
from factorization_dataloader import dataset_embedding, fact_net1, fact_net2, fact_net3
from utils_function import average_precision
import pandas as pd

## UTILITY FUNCTION

def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--input_dim", type = int, help = "dimension of input features/embedding", nargs='?', default=128)
    parser.add_argument("--model_name", type = str, help = "name of the factorization model",nargs='?')
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing embedding metadata")
    parser.add_argument("--model_weights", type = str, help = "path of the pre-trained model which will be used as a embedding extractor")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", nargs='?',default = "./output")
    parser.add_argument("--embedding_loc", type = str, help = "path to pre-computed embeddings for dataloader")
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--sub_embedding_dim", type = int, help = "value of subword embedding dimensions",nargs='?',default = 64)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    
    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])



def main():

    # For reproducibility

    torch.manual_seed(3112)
    torch.cuda.manual_seed(3112)
    torch.cuda.manual_seed_all(3112)
    np.random.seed(3112)
    random.seed(3112)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(3121)

    args = check_argv()

    # check whether the output path exists or not.

    isExist = os.path.exists(args.path_to_output)
  
    # Create a new directory for output because it does not exist 
    if not isExist:
        os.makedirs(args.path_to_output)
        print("The new directory for output is created to save subword embeddings!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size

    print("available device:",device)

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_data = dataset_embedding(
    metadata = os.path.join(args.metadata_file, "train_emb.csv"),
    embedding_mat = os.path.join(args.embedding_loc, "train_emb.pt")
    )
    val_data = dataset_embedding(
    metadata = os.path.join(args.metadata_file, "val_emb.csv"),
    embedding_mat=os.path.join(args.embedding_loc, "val_emb.pt")
    )
    test_data = dataset_embedding(
    metadata = os.path.join(args.metadata_file, "test_emb.csv"),
    embedding_mat = os.path.join(args.embedding_loc, "test_emb.pt")
    )
    
    print("length of training data:",len(train_data))
    print("length of validation data:",len(val_data))
    print("length of test data:",len(test_data))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    

    print("Calculating the available sub-words - bpe tokens - no. of classes from training set ...")

    token_list = []
    for _, (_, tokens, _, _, _) in enumerate(train_loader):
        token_list = token_list + tokens

    out = [item for t in token_list for item in t]
    dict_tokens = Counter(out)
    print("total tokens (number of sub-words):",len(dict_tokens))
    classes = dict_tokens.keys()
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(dict_tokens)

    def cal_precision(model, loader, device, distance):
        sub_embeddings, sub_words, words, unique_id = [], [], [], []
        model = model.eval()
        with torch.no_grad():
            for idx, (emb, tokens, _, word_name, sp_ch_id) in enumerate(loader):
                tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
                emb, tokens = emb.to(device), tokens.to(device)
            
                emb1, emb2, emb3 = model(emb)[0:3]
                l1 = tokens[0, :]
                l2 = tokens[1, :]
                l3 = tokens[2, :]

                sub_embeddings.append(emb1)
                sub_embeddings.append(emb2)
                sub_embeddings.append(emb3)
                
                sub_words.append(l1.cpu())
                sub_words.append(l2.cpu())
                sub_words.append(l3.cpu())
                unique_id.append(sp_ch_id*3)
                words.append(word_name*3)

        words = np.concatenate(words)       
        sub_words = np.concatenate(sub_words)
        unique_id = np.concatenate(unique_id)
        u_sub_words = np.unique(sub_words)
        sub_word2id = {v: k for k, v in enumerate(u_sub_words)}
        ids = [sub_word2id[w] for w in sub_words]
        sub_embeddings, ids = torch.cat(sub_embeddings,0), np.array(ids)
        # avg_precision,_ = average_precision(sub_embeddings.cpu(),ids, distance)
        avg_precision = 0.0
        return avg_precision, sub_embeddings, sub_words, unique_id, ids, words

    # Define the model
    # Define the model
    if args.model_name == "fact_net1":

        model = fact_net1(args.input_dim, args.sub_embedding_dim, num_classes)
        model = model.to(device)

    elif args.model_name == "fact_net2":
        model = fact_net2(args.input_dim, args.sub_embedding_dim, num_classes)
        model = model.to(device)

    elif args.model_name =="fact_net3":
        model = fact_net3(args.input_dim, args.sub_embedding_dim, num_classes)
        model = model.to(device)   

    checkpoint = torch.load(args.model_weights, map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("extracting subword embedding for test set")
    _, sub_emb, sub_words, unique_id, ids, words = cal_precision(model, test_loader, device, args.distance)
    # print("test average precision:", test_avg_precision)
    df_test = pd.DataFrame()
    df_test["unique_id"] = unique_id
    df_test["words"] = words
    df_test["ids"] = ids
    df_test["sub_words"] = np.vectorize(idx_to_class.get)(sub_words)
    torch.save(sub_emb, os.path.join(args.path_to_output, "test_sub_emb.pt"))
    df_test.to_csv(os.path.join(args.path_to_output,"test_sub_emb.csv"), index=False)

    print("extracting subword embedding for val set")
    _, sub_emb, sub_words, unique_id, ids, words = cal_precision(model, val_loader, device, args.distance)
    # print("val average precision:", val_avg_precision)

    df_val = pd.DataFrame()
    df_val["unique_id"] = unique_id
    df_val["words"] = words
    df_val["ids"] = ids
    df_val["sub_words"] = np.vectorize(idx_to_class.get)(sub_words)
    torch.save(sub_emb, os.path.join(args.path_to_output, "val_sub_emb.pt"))
    df_val.to_csv(os.path.join(args.path_to_output,"val_sub_emb.csv"), index=False)

    print("extracting subword embedding for train set")
    _, sub_emb, sub_words, unique_id, ids, words = cal_precision(model, train_loader, device, args.distance)
    # print("train average precision:", train_avg_precision)





    df_train = pd.DataFrame()
    df_train["unique_id"] = unique_id
    df_train["words"] = words
    df_train["ids"] = ids
    df_train["sub_words"] = np.vectorize(idx_to_class.get)(sub_words)
    torch.save(sub_emb, os.path.join(args.path_to_output, "train_sub_emb.pt"))
    df_train.to_csv(os.path.join(args.path_to_output,"train_sub_emb.csv"), index=False)

if __name__ == "__main__":
    main()