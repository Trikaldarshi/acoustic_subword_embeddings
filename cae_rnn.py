"""
train CAE-RNN network and save the model

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import argparse
import wandb
import datetime
import random
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import copy
from os import path
from awe_dataset_class import awe_dataset_pre_computed, cae_awe_dataset_pre_computed
from model_cae import model_cae
from utils_function import collate_fn, collate_fn_cae, save_checkpoints, load_checkpoints, average_precision, get_data





possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H","MFCC"]


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model_name", type=str, help = "name of the model for example, HUBERT_BASE", nargs='?', default = "HUBERT_BASE", choices = possible_models)
    parser.add_argument("--input_dim", type = int, help = "dimension of input features", nargs='?', default=768)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing paths of wave files, words, start point, duration \
      or SSL features metadata file")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", nargs='?',default = "./output")
    parser.add_argument("--root", type = str, help = "base path to librispeech dataset or None for SSL precomputed features", nargs='?',default = "None")
    parser.add_argument("--layer", type = int, help = "layer you want to extract, type mfcc for mfcc", nargs='?',default=0)
    parser.add_argument("--lr", type = float, help = "learning rate", nargs='?', default=0.001)
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--n_epochs", type = int, help = "number of epochs", nargs='?', default=10)
    parser.add_argument("--checkpoint_steps", type = int, help = "steps at which checkpoint will be saved",nargs='?',default = 2)
    parser.add_argument("--pre_compute", type = bool, help = "use pre computed features or not",nargs='?',default = True)
    parser.add_argument("--step_lr", type = int, help = "steps at which learning rate will decrease",nargs='?',default = 20)
    parser.add_argument("--embedding_dim", type = int, help = "value of embedding dimensions",nargs='?',default = 128)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    parser.add_argument("--opt", type = str, help = "optimizer", nargs='?', default = "adam", choices=["adam","sgd"])
    parser.add_argument("--loss", type = str, help = "loss function", nargs='?', default = "mse", choices=["mse","mae"])
    parser.add_argument("--hidden_dim", type = int, help = "rnn hidden dimension values", default=512)
    parser.add_argument("--rnn_type", type = str, help = " type or rnn, gru or lstm?", default="LSTM", choices=["GRU","LSTM"])
    parser.add_argument("--bidirectional", type = bool, help = " bidirectional rnn or not", default = True)
    parser.add_argument("--num_layers", type = int, help = " number of layers in rnn network, input more than 1", default=2) 
    parser.add_argument("--dropout", type = float, help = "dropout applied inside rnn network", default=0.2)
    parser.add_argument("--checkpoint_path", type = str, help = " path of the pre-trained model/checkpoint")
    parser.add_argument("--checkpoint_path_best", type = str, help = " path of the pre-trained model/checkpoint")

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])


def cal_precision(model,loader,device,distance):
  embeddings, words = [], []
  model = model.eval()
  with torch.no_grad():
    for _, (data,lens,word_name,_) in enumerate(loader):
      lens, perm_idx = lens.sort(0, descending=True)
      data = data[perm_idx]
      word_name = word_name[perm_idx]
      
      data, lens  = data.to(device), lens.to(device)


      emb = model.encoder(data, lens)
      embeddings.append(emb)
      words.append(word_name)
  words = np.concatenate(words)
  uwords = np.unique(words)
  word2id = {v: k for k, v in enumerate(uwords)}
  ids = [word2id[w] for w in words]
  embeddings, ids = torch.cat(embeddings,0), np.array(ids)
  avg_precision,_ = average_precision(embeddings.cpu(),ids, distance)

  return avg_precision


#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#

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


  ## read the parsed arguments

  args = check_argv()

  ## print the arguments you want to see:

  print("epochs:", args.n_epochs)
  print("batch_size:", args.batch_size)
  print("model name:", args.model_name)
  print("layer of the model:", args.layer)
  print("checkpoint steps:", args.checkpoint_steps)
  print("pre compute:",args.pre_compute)

  # Check whether the specified text/dataframe metadata file exists or not
  isExist = os.path.exists(args.metadata_file)

  if not isExist:
      print(args.metadata_file)
      print("provide the correct path for the text/dataframe file having list of wave files")
      sys.exit(1)
  if args.root=="None":
    args.root = ""
  else:
    isExist = os.path.exists(args.root)

    if not isExist:
        print("provide the correct path for the librispeech dataset")
        sys.exit(1)

  # Check whether the specified output path exists or not
  isExist = os.path.exists(args.path_to_output)
  
  # Create a new directory for output if it does not exist 

  if not isExist:
      os.makedirs(args.path_to_output)
      print("The new directory for output is created!")

  ## create a unique output storage location with argument files

  now = str(datetime.datetime.now())
  os.makedirs(path.join(args.path_to_output,now))
  args.path_to_output = path.join(args.path_to_output,now)

  with open(path.join(args.path_to_output,'config.txt'), 'w') as f:
    for key, value in vars(args).items(): 
            f.write('--%s=%s\n' % (key, value))
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  batch_size = args.batch_size

  print("available device:",device)

  if device == "cuda":
      num_workers = 1
      pin_memory = True
  else:
      num_workers = 0
      pin_memory = False


  if args.pre_compute:

    print("using pre-computed features", args.model_name)
    
    train_data = cae_awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="train"
    )
    val_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="val"
    )
    test_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="test"
    )
  else:

    train_data, val_data, test_data = get_data(args.model_name, args.root, args.device, args.metadata_file, args.layer)



  print("length of training data:",len(train_data))
  print("length of validation data:",len(val_data))
  print("length of test data:",len(test_data))
  
  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=collate_fn_cae,
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
      collate_fn=collate_fn,
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
      collate_fn=collate_fn,
      drop_last = False,
      num_workers=num_workers,
      pin_memory=pin_memory,
      worker_init_fn=seed_worker,
      generator=g
  )

  ## Define the model

  model = model_cae(args.input_dim, args.hidden_dim, args.embedding_dim, args.rnn_type, args.bidirectional, args.num_layers, args.dropout)
  model = model.to(device)


  model_description = '_'.join([args.model_name, str(args.embedding_dim)])


  # Check whether the specified checkpoint/pre-trained model exists or not
  isCheckpoint = os.path.exists(args.checkpoint_path)

  if isCheckpoint:
    print("loading the pre-trained... hold on.....")
    PATH = args.checkpoint_path
    PATH_BEST = args.checkpoint_path_best
    print(PATH)
    print(PATH_BEST)
  else:
    print("no checkpoint given, starting training from scratch!")
    PATH = path.join(args.path_to_output, model_description + ".pt")
    PATH_BEST = path.join(args.path_to_output, model_description + "_BEST.pt")


  def train_model(model, train_load, val_load, n_epochs):

    if args.opt=="sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.5)
    if args.loss=="mse":
      criterion = nn.MSELoss().to(device)
    else:
      criterion = nn.L1Loss().to(device)

    if isCheckpoint==True:
      print("loading the saved checkpoint for training:")
      print("best model:")
      model_best, _, _, best_epoch, history_best = load_checkpoints(model, optimizer, scheduler, PATH_BEST, device)
      print("recent checkpoint:")
      model, optimizer, scheduler, base_epoch, history = load_checkpoints(model, optimizer, scheduler, PATH, device)
      optimizer.param_groups[0]['capturable'] = True # Error: assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors.
      scheduler.step()
      base_epoch += 1
      best_model_wts = copy.deepcopy(model_best.state_dict())
      best_ap = history_best['val_avg_precision'][-1]

    else:
      history = dict(train=[], val=[], val_avg_precision=[])
      base_epoch = 1
      best_model_wts = copy.deepcopy(model.state_dict())
      best_ap = 0.0
    print("training starting at epoch - ", base_epoch)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(base_epoch, n_epochs + 1):
      model = model.train()

      train_losses = []
      for _, (x, lens_x,_,_, y, lens_y,_,_) in enumerate(train_load):
        optimizer.zero_grad()
        x, lens_x, y, lens_y = x.to(device), lens_x.to(device), y.to(device), lens_y.to(device)
        lens_x, perm_idx = lens_x.sort(0, descending=True)
        x = x[perm_idx]
        y = y[perm_idx]
        lens_y = lens_y[perm_idx]

        seq_pred = model(x, lens_x, lens_y)
        loss = criterion(seq_pred, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

      val_losses = []
      model = model.eval()
      with torch.no_grad():
        for _, (data,lens,word_name,_) in enumerate(val_load):
          data, lens = data.to(device), lens.to(device)
          lens, perm_idx = lens.sort(0, descending=True)
          data = data[perm_idx]

          seq_pred = model(data, lens, lens)

          loss = criterion(seq_pred, data) # data[0] will give s1-[20,768]; s2-[2,768]; s3-[3,768]:--------> [25,768]
                                                      # data_packed.data will return the [[25,768]] thing only
          val_losses.append(loss.item())
        

      # train_avg_precision = cal_precision(model, train_load, device, args.distance)
      val_avg_precision = cal_precision(model, val_load, device, args.distance)
      train_loss = np.mean(train_losses)
      val_loss = np.mean(val_losses)

      history['train'].append(train_loss)
      history['val'].append(val_loss)
      history["val_avg_precision"].append(val_avg_precision)

      if val_avg_precision > best_ap:
        best_ap = val_avg_precision
        best_model_wts = copy.deepcopy(model.state_dict())
        print("checkpoint saved for best epoch for average precision")
        save_checkpoints(epoch,model,optimizer,scheduler,history,PATH_BEST)
        best_epoch = epoch
      
      if epoch % args.checkpoint_steps == 0:
        print("checkpoint logging :")
        save_checkpoints(epoch,model,optimizer,scheduler,history,PATH)


      print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} ; best epoch {best_epoch} val avg precision {val_avg_precision}')
      print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))


      scheduler.step()

    model.load_state_dict(best_model_wts)

    return model.eval(), history


  model, _ = train_model(
    model, 
    train_loader, 
    val_loader, 
    n_epochs=args.n_epochs
  )


  test_avg_precision = cal_precision(model, test_loader, device, args.distance)

  print("average precision on test set:", test_avg_precision)
  print(" We are done! Bye Bye. Have a nice day!")


if __name__ == "__main__":
    main()

