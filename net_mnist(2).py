from __future__ import print_function
import argparse
import numpy.random as npr
import time
import torch.optim as optim
from torchvision import datasets, transforms
import datetime
import json
import random
import math
from math import pow, acos, sqrt
from itertools import chain
from collections import Counter
import torch
from torch import nn
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.datasets import make_blobs, make_regression, make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample
import copy
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn import datasets
import asyncio
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import os
from PIL import Image
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device = torch.device("cpu")


def build_new_train_dataloader(visited_indexes, last_train_dataloader, train_batch_size, shuffle, n_features):
  result_train_X = []
  result_train_y = []
  last_index = 0

  set_seed()
  for inde_batch, (train_data_x, train_data_y) in enumerate(last_train_dataloader):
    for index, (train_data_x_i, train_data_y_i) in enumerate(zip(train_data_x, train_data_y)): 
      current_index = index + last_index
      if current_index in visited_indexes:
        result_train_X.append(train_data_x_i.tolist())
        result_train_y.append(train_data_y_i.tolist())
    last_index += train_data_x.shape[0]

  if len(result_train_X) != len(result_train_y): 
    print(f"len_result_train_X: {len(result_train_X)}, len_result_train_y: {len(result_train_y)}")
    print(result_train_X)
    print(result_train_y)
  
  result_train_X, result_train_y = np.array(result_train_X), np.array(result_train_y)
  result_train_X, result_train_y = result_train_X.astype(float), result_train_y.astype(float)
  new_train_dataloader = DataLoader(TensorDataset(torch.tensor(np.array([])), torch.tensor(np.array([]))))
  result_train_dataset = TensorDataset(torch.tensor(result_train_X).to(torch.float32), torch.tensor(result_train_y).to(torch.float32))
  if len(result_train_dataset) > 0: new_train_dataloader = DataLoader(result_train_dataset, batch_size = train_batch_size, shuffle = shuffle)
  return new_train_dataloader


def check_full(min_max_data_nums_per_class):
  global class_nums_counter
  for k, v in class_nums_counter.items():
    if v < min_max_data_nums_per_class[1]:
      return False
  return True


def filter_train_data_by_angle(sorted_result_angle, threshold_occupation, min_max_data_nums_per_class, current_epoch_all_train_y,\
                    visited_indexes, type_nums_counter, new_train_dataloader, epoch, batch_index):
  global class_nums_counter
  # sorted_result_angle:
  # v1. (angle1, index_other1, index_me), (angle2, index_other2, index_me) ... sorted angle


  me_index = sorted_result_angle[-1][2]

  me_real_class = current_epoch_all_train_y[me_index]
  
  current_max_angle = sorted_result_angle[-1][0]
  current_min_angle = sorted_result_angle[0][0]
  current_median_angle = sorted_result_angle[len(sorted_result_angle) // 2][0]

  current_min_threshold_angle = current_min_angle + (current_median_angle - current_min_angle) / threshold_occupation
  current_max_threshold_angle = current_max_angle - (current_max_angle - current_median_angle) / threshold_occupation

  for i in range(len(sorted_result_angle)):
    current_result_angle = sorted_result_angle[i][0]
    current_result_index = sorted_result_angle[i][1]
    current_result_real_class = current_epoch_all_train_y[current_result_index]
    if current_result_angle > current_min_threshold_angle: break
    if current_result_real_class != me_real_class:
      # ??????????????????????????????????????????????????????????????????????????????????????????????????????
      if class_nums_counter[me_real_class] < min_max_data_nums_per_class[1] and me_index not in visited_indexes:
        class_nums_counter[me_real_class] += 1
        visited_indexes.add(me_index)
        type_nums_counter[0] += 1
      if class_nums_counter[current_result_real_class] < min_max_data_nums_per_class[1] and current_result_index not in visited_indexes:
        class_nums_counter[current_result_real_class] += 1
        visited_indexes.add(current_result_index)
        type_nums_counter[0] += 1
    else:
      # ???????????????????????????????????????????????????me_index??????????????????????????????????????????
      if class_nums_counter[me_real_class] < min_max_data_nums_per_class[1] and me_index not in visited_indexes:
        class_nums_counter[me_real_class] += 1
        visited_indexes.add(me_index)
        type_nums_counter[1] += 1

  
  for i in range(len(sorted_result_angle) - 1, 0, -1):
    current_result_angle = sorted_result_angle[i][0]
    current_result_index = sorted_result_angle[i][1]
    current_result_real_class = current_epoch_all_train_y[current_result_index]
    if current_result_angle < current_max_threshold_angle: break
    if current_result_real_class != me_real_class:
      # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
      if class_nums_counter[me_real_class] < min_max_data_nums_per_class[1] and me_index not in visited_indexes:
        class_nums_counter[me_real_class] += 1
        visited_indexes.add(me_index)
        type_nums_counter[2] += 1
      if class_nums_counter[current_result_real_class] < min_max_data_nums_per_class[1] and current_result_index not in visited_indexes:
        class_nums_counter[current_result_real_class] += 1
        visited_indexes.add(current_result_index)
        type_nums_counter[2] += 1
    else:
      # ?????????????????????????????????????????????????????????????????????????????????????????????????????????
      if class_nums_counter[me_real_class] < min_max_data_nums_per_class[1] and me_index not in visited_indexes:
        class_nums_counter[me_real_class] += 1
        visited_indexes.add(me_index)
        type_nums_counter[3] += 1
      if class_nums_counter[current_result_real_class] < min_max_data_nums_per_class[1] and current_result_index not in visited_indexes:
        class_nums_counter[current_result_real_class] += 1
        visited_indexes.add(current_result_index)
        type_nums_counter[3] += 1


  # ????????????????????????????????????
  if class_nums_counter[me_real_class] < min_max_data_nums_per_class[0] and me_index not in visited_indexes:
    class_nums_counter[me_real_class] += 1
    visited_indexes.add(me_index)
    type_nums_counter[4] += 1



def filter_train_data_by_angle_one_batch(sorted_result_angle_one_batch, threshold_occupation, min_max_data_nums_per_class, current_epoch_all_train_y,\
                    visited_indexes, type_nums_counter, new_train_dataloader, epoch, batch_index):
  [filter_train_data_by_angle(sorted_result_angle_one_batch[i],\
    threshold_occupation, min_max_data_nums_per_class, current_epoch_all_train_y, visited_indexes, type_nums_counter, new_train_dataloader, epoch, batch_index)\
    for i in range(len(sorted_result_angle_one_batch))]


def update_dataloader(threshold_epoch, current_test_accuracy, type_nums_counter, visited_indexes, current_epoch_accuracy_detail_lst):
  global start_new_dataloader_min_nums, last_selected_train_x_indexes, intepreter_selected_indexes, threshold_test_accuracy_to_build_new_data, epoch_accuracy_matrix,\
   new_train_dataloader, n_features, filter_button
  if threshold_epoch != None and current_test_accuracy >= threshold_test_accuracy_to_build_new_data and filter_button == True:
    to_build_new_data = False
    if len(visited_indexes) < start_new_dataloader_min_nums:
      filter_button = False
      print(f"not enough filter data: {len(visited_indexes)} using last_index instead, {len(last_selected_train_x_indexes)}")
      visited_indexes = set([i for i in range(len(last_selected_train_x_indexes))]) # not use all_new_train_data_index = last_selected_train_x_indexes
    else:
      filter_button = False
      to_build_new_data = True
      print(f"len(new_visited_indexes) {len(visited_indexes)}")
      # print(f"new_visited_indexes: {visited_indexes}")
    
    for k, v in type_nums_counter.items():
      print(f"{intepreter_selected_indexes[k]} {v}")
    print(class_nums_counter)
    
    epoch_accuracy_matrix = np.array(epoch_accuracy_matrix)
    print(f"epoch_accuracy_matrix shape: {epoch_accuracy_matrix.shape}")
    epoch_accuracy_matrix = epoch_accuracy_matrix.T[list(visited_indexes)].T
    epoch_accuracy_matrix = epoch_accuracy_matrix.tolist()

    current_epoch_accuracy_detail_lst = np.array(list(chain(*current_epoch_accuracy_detail_lst)))
    current_epoch_accuracy_detail_lst = current_epoch_accuracy_detail_lst[list(visited_indexes)]

    epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
    
    current_epoch_accuracy_detail_lst = current_epoch_accuracy_detail_lst.tolist()
    print(f"append finished, epoch_accuracy_matrix shape: {np.array(epoch_accuracy_matrix).shape}\n")

    if to_build_new_data: new_train_dataloader = build_new_train_dataloader(visited_indexes, new_train_dataloader, train_batch_size = 64, shuffle = True, n_features = n_features)
    last_selected_train_x_indexes = visited_indexes
    # if epoch % 3 == 0: threshold_angle *= 0.985
  else:
    current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_lst))
    epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
    print("\n")


def update_threshold_epoch(acc_train_list, loss_train_list):
  global threshold_window, threshold_train_acc, threshold_stable_acc, threshold_stable_loss, threshold_epoch
  if len(acc_train_list) > threshold_window and threshold_epoch == None:
    temp_acc_train_list, temp_loss_train_list = sorted(acc_train_list[- threshold_window - 1: -1]), sorted(loss_train_list[- threshold_window - 1: -1])
    min_acc_train, max_acc_train, mediean_acc_train = temp_acc_train_list[0], temp_acc_train_list[-1], sum(temp_acc_train_list) / threshold_window
    min_loss_train, max_loss_train = temp_loss_train_list[0], temp_loss_train_list[-1]
    delta_acc_train = max_acc_train - min_acc_train
    delta_loss_train = max_loss_train - min_loss_train
    print(f"delta_acc_train: {delta_acc_train}\ndelta_loss_train: {delta_loss_train}\nmin_loss_train: {min_loss_train}\nmax_loss_train: {max_loss_train}\nmin_acc_train: {min_acc_train}\nmax_acc_train: {max_acc_train}\nmediean_acc_train: {mediean_acc_train}\n")
    if mediean_acc_train > threshold_train_acc:
      threshold_epoch = epoch
      print("acc")
    elif delta_acc_train < threshold_stable_acc and delta_loss_train < threshold_stable_loss:
      threshold_epoch = epoch
      print("delta")
  

def calculate_angle_update_visited_index(threshold_epoch, last_index, current_epoch_accuracy_detail_lst_flattern, batch_index, tot_batches, train_x_shape_0,\
                      current_epoch_all_train_y, visited_indexes, type_nums_counter):
    global epoch_accuracy_matrix, batch_to_filter, filter_button, sorted_result_angles
    if threshold_epoch != None and last_index != 0 and check_full(min_max_data_nums_per_class) == False and filter_button == True:
      epoch_accuracy_matrix = np.array(epoch_accuracy_matrix)
      epoch_accuracy_matrix_ne = copy.deepcopy(epoch_accuracy_matrix.T[:last_index, :]) 
      epoch_accuracy_matrix = epoch_accuracy_matrix.tolist()
      epoch_accuracy_matrix_ne = np.insert(epoch_accuracy_matrix_ne, epoch_accuracy_matrix_ne.shape[1],\
                      values = current_epoch_accuracy_detail_lst_flattern[:last_index], axis = 1)

      current_upper_matrix = torch.tensor(np.array(epoch_accuracy_matrix_ne[:last_index])).half().to(device) #fp16
      current_uppper_train_y = np.array(current_epoch_all_train_y)[:last_index]
      
      current_compare_matrix = torch.tensor(np.array(epoch_accuracy_matrix_ne[last_index - train_x_shape_0:last_index])).half().to(device)
      current_compare_train_y = np.array(current_epoch_all_train_y)[last_index - train_x_shape_0:last_index]
      
      sorted_result_angles_one_batch = torch.acos(torch.cosine_similarity(current_compare_matrix.unsqueeze(1), current_upper_matrix.unsqueeze(0),dim=-1))
      sorted_result_angles_one_batch *= 180 / math.pi

      sorted_result_angles_one_batch = sorted_result_angles_one_batch.cpu().numpy()
      l1, l2 = sorted_result_angles_one_batch.shape

  
      sorted_result_angles_one_batch = [sorted([(sorted_result_angles_one_batch[i][other_index], other_index, last_index - train_x_shape_0 + i)\
                                for other_index in range(l2)], key = lambda x: x[0]) for i in range(l1)]

      # shape: batch_size * Angle_list, each element in Angle_list: (angle, other_index, me_index), sorted                                                                                     
      sorted_result_angles.append(sorted_result_angles_one_batch)

      # ??????????????????????????????
      del sorted_result_angles_one_batch
      del current_upper_matrix
      del current_compare_matrix
      del epoch_accuracy_matrix_ne
      del current_uppper_train_y
      del current_compare_train_y

      if (batch_to_filter != 0 and batch_index % batch_to_filter == 0) or batch_index == tot_batches:
        print(f"current_batch_index: {batch_index}, tot_bacthes: {tot_batches}")
        
        if batch_index == tot_batches:
          print("last batch!")

        [filter_train_data_by_angle_one_batch(sorted_result_angles_one_batch,\
          threshold_occupation, min_max_data_nums_per_class, current_epoch_all_train_y, visited_indexes, type_nums_counter, new_train_dataloader, epoch, batch_index)\
          for sorted_result_angles_one_batch in sorted_result_angles]

        print(f"current len visited_indexes: {len(visited_indexes)}. max_new_data: {min_max_data_nums_per_class[1] * 10}.")
        
        del sorted_result_angles
        sorted_result_angles = [] #?????????????????????????????????????????????

def train_one_epoch(model, device, optimizer):
    global intepreter_selected_indexes, threshold_epoch, sorted_result_angles, filter_button

    model.train()

    current_epoch_all_train_y = []
    current_epoch_accuracy_detail_lst = []
    sorted_result_angles = []
    
    last_index, tot_train_correct, tot_train_nums, tot_train_loss = 0, 0, 0, 0
    type_nums_counter = {i: 0 for i in range(len(intepreter_selected_indexes))}
    visited_indexes = set()
    tot_batches = new_train_dataloader.dataset.__len__() // new_train_dataloader.batch_size
    batch_cnt = 0
    set_seed()
    for batch_index, (train_x, train_y) in enumerate(new_train_dataloader):
      batch_cnt += 1
      train_x, train_y = train_x.to(device), train_y.to(device)

      optimizer.zero_grad()
      
      outputs = model(train_x)
      train_y = train_y.type(torch.LongTensor)
      train_y = train_y.to(device)
      loss = criterion(outputs, train_y)
      _, pred_y = torch.max(outputs.data, 1)

      result = torch.eq(pred_y, train_y)
      tot_train_correct += result.float().sum()
      last_index += train_x.shape[0]
      tot_train_nums += train_x.shape[0]

      current_epoch_accuracy_detail_lst.append(result.tolist())
      current_epoch_accuracy_detail_lst_flattern = np.array(list(chain(*current_epoch_accuracy_detail_lst)))
      
      for i, train_y_i in enumerate(train_y):
        current_epoch_all_train_y.append(train_y_i.item())
      
      if filter_button == True:
        calculate_angle_update_visited_index(threshold_epoch, last_index, current_epoch_accuracy_detail_lst_flattern, batch_index, tot_batches, train_x.shape[0],\
                            current_epoch_all_train_y, visited_indexes, type_nums_counter)

      loss = loss.mean()
      loss.backward()
      optimizer.step()
      tot_train_loss += loss.detach()
    
    current_train_accuracy = tot_train_correct.item() / tot_train_nums
    current_train_loss = tot_train_loss.item() / batch_cnt
    
    return current_train_accuracy, current_train_loss, current_epoch_accuracy_detail_lst, visited_indexes, type_nums_counter

def test_one_epoch(model, device, test_dataloader):
  model.eval()
  correct = 0
  tot = 0
  tot_loss = 0
  batch_cnt = 0
  with torch.no_grad():
    # torch.manual_seed(42)
    for index_batch, (test_x, test_y) in enumerate(test_dataloader):
      batch_cnt += 1
      test_x, test_y = test_x.to(device), test_y.to(device)
      logits = model(test_x)
      _, pred_y = torch.max(logits.data, 1)
      correct += torch.eq(pred_y, test_y).float().sum()
      tot += test_x.shape[0]
      loss = criterion(logits, test_y.type(torch.LongTensor).to(device))
      tot_loss += loss.mean()

  current_test_accuracy = correct.item() / tot
  current_test_loss = tot_loss / batch_cnt
  return current_test_accuracy, current_test_loss
  
# Setup basic CNN model
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))

    # if args.no_dropout: # no_dropout = false
    #     x = F.relu(F.max_pool2d(self.conv2(x), 2))
    # else:
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))

    # if not args.no_dropout:
    x = F.dropout(x, training=self.training)

    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

test_batch_size = 32
batch_size = 64
epochs = 200
lr = 0.01
momentum = 0.5

# Set appropriate devices
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")
kwargs = {'num_workers': 0, 'pin_memory': True} # {'num_workers': 1, 'pin_memory': True}


# Set random seed for initialization
def set_seed(seed = 1):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
  npr.seed(seed)

set_seed()

# Setup transforms
all_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
]
transform = transforms.Compose(all_transforms)
trainset = torchvision.datasets.MNIST(
    root=r'C:\\Users\\GM\Desktop\\liuzhengchang', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(
    root=r'C:\\Users\\GM\Desktop\\liuzhengchang', train=False, download=True, transform=transform)
new_train_dataloader = DataLoader(dataset = trainset, 
                  batch_size = batch_size, 
                  num_workers = kwargs["num_workers"],
                  pin_memory = kwargs["pin_memory"],
                  shuffle = True)
test_dataloader = DataLoader(dataset = testset, 
                  batch_size = test_batch_size,
                  num_workers = kwargs["num_workers"],
                  pin_memory = kwargs["pin_memory"],
                  shuffle = True)

# Setup model and optimizer
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


# Setup loss
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)

n_samples, *n_features = new_train_dataloader.dataset.data.shape
epoch_train_dataloader = []
epoch_accuracy_matrix = []
sorted_result_angles = []
intepreter_selected_indexes = {\
                  0: "??????????????????????????????????????????????????????????????????????????????????????????????????????: ",\
                  1: "???????????????????????????????????????????????????me_index?????????????????????????????????????????????",\
                  2: "????????????????????????????????????????????????????????????????????????????????????????????????????????????",\
                  3: "????????????????????????????????????????????????????????????????????????????????????????????????????????????",\
                  4: "???????????????????????????????????????"}
threshold_window, threshold_stable_acc, threshold_stable_loss, threshold_train_acc = 3, 1e-3, 1e-3, 0.85 
acc_train_list, acc_test_list, loss_train_list, loss_test_list = [], [], [], []
threshold_epoch = None
last_selected_train_x_indexes = set([i for i in range(new_train_dataloader.dataset.__len__())])
class_nums_counter = {class_name:0 for class_name in range(10)}
min_max_data_nums_per_class = [10, 1200]
start_new_dataloader_min_nums = 100
threshold_test_accuracy_to_build_new_data = 0.600
threshold_occupation = 2
batch_to_filter = 5
filter_button = True
original_new_train_dataloader = new_train_dataloader
last_new_train_dataloader = new_train_dataloader

print("20!")
print("??????????????????????????????-??????????????????????????????")
torch.cuda.empty_cache()
gc.collect()
print(f"train_dataloader len: {new_train_dataloader.dataset.__len__()}")
acc_train_list, acc_test_list, loss_train_list, loss_test_list = [], [], [], []
for epoch in range(epochs):
  current_train_accuracy, current_train_loss, current_epoch_accuracy_detail_lst, visited_indexes, type_nums_counter = train_one_epoch(model, device, optimizer)
  acc_train_list.append(current_train_accuracy)
  loss_train_list.append(current_train_loss)

  current_test_accuracy, current_test_loss = test_one_epoch(model, device, test_dataloader)
  acc_test_list.append(current_test_accuracy)
  loss_test_list.append(current_test_loss)
  
  last_new_train_dataloader = new_train_dataloader

  if filter_button == True: 
    update_dataloader(threshold_epoch, current_test_accuracy, type_nums_counter, visited_indexes, current_epoch_accuracy_detail_lst)
  if threshold_epoch == None:
    update_threshold_epoch(acc_train_list, loss_train_list)
  print(f"epoch: {epoch}, acc_train: {acc_train_list[-1]}, loss_train: {loss_train_list[-1]}, acc_test: {acc_test_list[-1]}, loss_test: {loss_test_list[-1]}")


print(f"acc_test_list:\n{acc_test_list}\n")
print(f"acc_train_list:\n{acc_train_list}\n")
print(f"loss_train_list:\n{loss_train_list}\n")
print(f"loss_test_list:\n{loss_test_list}\n")



print("????????????")
filter_button = False
threshold_epoch = None
new_train_dataloader = last_new_train_dataloader # epoch_train_dataloader[-1]
torch.cuda.empty_cache()
gc.collect()
acc_train_list, acc_test_list, loss_train_list, loss_test_list = [], [], [], []
print(f"train_dataloader len: {new_train_dataloader.dataset.__len__()}")
for epoch in range(epochs):
  current_train_accuracy, current_train_loss, current_epoch_accuracy_detail_lst, visited_indexes, type_nums_counter = train_one_epoch(model, device, optimizer)
  acc_train_list.append(current_train_accuracy)
  loss_train_list.append(current_train_loss)

  current_test_accuracy, current_test_loss = test_one_epoch(model, device, test_dataloader)
  acc_test_list.append(current_test_accuracy)
  loss_test_list.append(current_test_loss)

  if filter_button == True: 
    update_dataloader(threshold_epoch, current_test_accuracy, type_nums_counter, visited_indexes, current_epoch_accuracy_detail_lst)
  if threshold_epoch == None:
    update_threshold_epoch(acc_train_list, loss_train_list)
  print(f"epoch: {epoch}, acc_train: {acc_train_list[-1]}, loss_train: {loss_train_list[-1]}, acc_test: {acc_test_list[-1]}, loss_test: {loss_test_list[-1]}")


print(f"acc_test_list:\n{acc_test_list}\n")
print(f"acc_train_list:\n{acc_train_list}\n")
print(f"loss_train_list:\n{loss_train_list}\n")
print(f"loss_test_list:\n{loss_test_list}\n")



print("????????????")
filter_button = False
threshold_epoch = None
new_train_dataloader = original_new_train_dataloader
torch.cuda.empty_cache()
gc.collect()
print(f"train_dataloader len: {new_train_dataloader.dataset.__len__()}")
acc_train_list, acc_test_list, loss_train_list, loss_test_list = [], [], [], []
for epoch in range(epochs):
  current_train_accuracy, current_train_loss, current_epoch_accuracy_detail_lst, visited_indexes, type_nums_counter = train_one_epoch(model, device, optimizer)
  acc_train_list.append(current_train_accuracy)
  loss_train_list.append(current_train_loss)

  current_test_accuracy, current_test_loss = test_one_epoch(model, device, test_dataloader)
  acc_test_list.append(current_test_accuracy)
  loss_test_list.append(current_test_loss)

  if filter_button == True: 
    update_dataloader(threshold_epoch, current_test_accuracy, type_nums_counter, visited_indexes, current_epoch_accuracy_detail_lst)
  if threshold_epoch == None:
    update_threshold_epoch(acc_train_list, loss_train_list)
  print(f"epoch: {epoch}, acc_train: {acc_train_list[-1]}, loss_train: {loss_train_list[-1]}, acc_test: {acc_test_list[-1]}, loss_test: {loss_test_list[-1]}")


print(f"acc_test_list:\n{acc_test_list}\n")
print(f"acc_train_list:\n{acc_train_list}\n")
print(f"loss_train_list:\n{loss_train_list}\n")
print(f"loss_test_list:\n{loss_test_list}\n")
