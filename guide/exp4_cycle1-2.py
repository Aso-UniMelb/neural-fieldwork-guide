import sys
import os
import random
import re
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import time
import numpy as np

# uncomment the following lines to export the heatmap:
# import matplotlib
# matplotlib.use('TkAgg',force=True)
# from matplotlib import pyplot as plt
# import seaborn as sns
  
#==================== load oracle data ====================
oracle_knowledge = {}
working_dir = ''

def load_oracle_knowledge(data_file):
  global oracle_knowledge
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
      lemma, form, tags = line.strip().split('\t')
      k = f'{lemma}@{tags}'
      oracle_knowledge[k] = form
  
# get the form from oracle
def get_form(lemma, tags):
  k = f'{lemma}@{tags}'
  if k in oracle_knowledge:
    form = oracle_knowledge[k]
    with open(f'{working_dir}/oracle_stat.tsv', 'a', encoding='utf-8') as w:
      result = 0
      predicted = ''
      w.write(f'{result}\t{predicted}\t{form}\t{lemma}\t{tags}\n')
    return form
  else:
      return ''

def main(lang, data_path, work_path, num_samples):
  #==================== options ====================
  trn_dev_split_ratio = [0.45, 0.55]
  batch_size = 1
  learning_rate = 0.005
  num_epochs = 10
  hidden_dim = 128
  n_layers = 1
  #==================== startup ====================
  global working_dir
  working_dir = f'{work_path}' 
  path = f'{working_dir}/{lang}'
  data_file = f'{data_path}/{lang}.tsv'
  if not os.path.exists(working_dir):
    os.makedirs(working_dir)
  load_oracle_knowledge(data_file)
  #==================== Full paradigms ====================
  # read the list of lemmas for getting full paradigms
  train_lemmas = []
  with open(f'{data_path}/{lang}-lemmas.txt', 'r', encoding='utf-8') as f:
    for line in f:
      train_lemmas.append(line.strip())
  
  # read the original data file
  # extract full paradigms from the full data
  paradigms = {}
  unique_tagsets = set()
  _pool = []
  samples = []
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
      lemma, form, tags = line.strip().split('\t')
      _pool.append(f'{lemma}\t{tags}\n')
      if lemma in train_lemmas:
        samples.append(f'{lemma}\t{tags}\n')
        if lemma not in paradigms:
          paradigms[lemma] = []
          paradigms[lemma].append([lemma, 'LEMMA'])
          unique_tagsets.add('LEMMA')
        paradigms[lemma].append([form, tags])
        unique_tagsets.add(tags)      
  
  with open(f'{path}.oracle', 'w', encoding='utf-8') as w:
    for line in samples:
      # Remove selected instances from the original list
      _pool.remove(line)
      # getting the forms from oracle
      lemma, tags = line.strip().split('\t')
      form = get_form(lemma, tags)
      w.write(f'{lemma}\t{form}\t{tags}\n')
  # Write the remaining instances back to the pool file
  with open(f'{path}.pool', 'w', encoding='utf-8') as f:
    f.writelines(_pool)

  # for each paradigm, generate train data set by generating all combinations of 2 forms
  paradigms_data = []
  for lemma, forms in paradigms.items():
    for i in range(len(forms)):
      for j in range(len(forms)):
        if i != j:
          src_form, src_tag, trgt_form, trgt_tag = forms[i][0], forms[i][1], forms[j][0], forms[j][1]
          r = f'{src_form}\t{src_tag}\t{trgt_tag}\t{trgt_form}\n'
          paradigms_data.append(r)

  # split random data into train, dev, test
  split = [int(len(paradigms_data) * trn_dev_split_ratio[0]), int(len(paradigms_data) * trn_dev_split_ratio[1])]
  random.shuffle(paradigms_data)
  with open(f'{path}.1.trn', 'w', encoding='utf-8') as trn:
    for i in range(split[0]):
      trn.write(paradigms_data[i])
  with open(f'{path}.1.dev', 'w', encoding='utf-8') as dev:
    for i in range(split[0], split[1]): 
      dev.write(paradigms_data[i])
  with open(f'{path}.1.tst', 'w', encoding='utf-8') as tst:  
    for i in range(split[1], len(paradigms_data)):
      tst.write(paradigms_data[i])
  #==================== train ==================== 

  f_train = open(f'{path}.1.trn', 'r', encoding='UTF8').read().splitlines()
  f_dev = open(f'{path}.1.dev', 'r', encoding='UTF8').read().splitlines()
  f_test = open(f'{path}.1.tst', 'r', encoding='UTF8').read().splitlines()

  len_tagsets = len(unique_tagsets) + 1
  heatmap_acc = np.empty((len_tagsets, len_tagsets))
  heatmap_acc[:] = np.nan
  heatmap_conf = np.empty((len_tagsets, len_tagsets))
  heatmap_conf[:] = np.nan

  for tag in unique_tagsets:
    data_train = []
    data_dev = []
    data_test = []
    for line in f_train:
      l = line.split('\t')
      if l[1] == tag:
        data_train.append(l)
    for line in f_dev:
      l = line.split('\t')
      if l[1] == tag:
        data_dev.append(l)  
    for line in f_test:
      l = line.split('\t')
      if l[1] == tag:
        data_test.append(l)
    # extract unique chars in lemmas and forms
    words = [[item[0], item[3]] for item in (data_train + data_dev)]
    all_chars = sorted(set(''.join([char for row in words for char in row])))
    all_chars.append('.')
    all_chars.append('$') # unknown
    ch_to_idx = {ch: idx for idx, ch in enumerate(all_chars)}
    idx_to_ch = {idx: ch for idx, ch in enumerate(all_chars)}
    max_length = 30 #max(len(word) for line in words for word in line)
    # extract unique tags
    # tags_set =  [[item[1], item[2]] for item in (data_train + data_dev)]
    unique_tags =  sorted(set(unique_tagsets))
    unique_tags.append('$') # unknown
    tg_to_idx = {t: idx for idx, t in enumerate(unique_tags)}
    idx_to_tg = {idx: t for idx, t in enumerate(unique_tags)}

    def encode(seq, lookup):
      return [lookup.get(item, lookup['$']) for item in seq]

    def encode_tag(tag):
      return tg_to_idx.get(tag, tg_to_idx['$'])

    def encode_word(word):
      return encode(word.ljust(max_length + 1, '.'), ch_to_idx)

    # load data into pyTorch dataloader
    def fill_dataloader(data, batch_size):
      encoded_trgt_forms = []
      encoded_trgt_tags = []
      encoded_src_forms = []
      encoded_src_tags = []
      for row in data:
        src_form, src_tag, trgt_tag, trgt_form = row
        encoded_trgt_forms.append(encode_word(trgt_form))
        encoded_src_tags.append([encode_tag(trgt_tag)])
        encoded_src_forms.append(encode_word(src_form))
        encoded_trgt_tags.append([encode_tag(src_tag)])
      dataset = TensorDataset(torch.tensor(encoded_src_forms), torch.tensor(encoded_src_tags), torch.tensor(encoded_trgt_tags), torch.tensor(encoded_trgt_forms))
      return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    train_dataloader = fill_dataloader(data_train, batch_size)
    dev_dataloader = fill_dataloader(data_dev, batch_size)
    test_dataloader = fill_dataloader(data_test, batch_size)

    # Define BiLSTM model
    class BiLSTM(nn.Module):
      def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = True
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

      def forward(self, src_form, src_tag, trgt_tag):
        embedded_src_form = self.embedding(src_form)
        embedded_src_tag = self.embedding(src_tag)
        embedded_trgt_tag = self.embedding(trgt_tag)
        embedded = embedded_src_form + embedded_src_tag + embedded_trgt_tag
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(lstm_out))
        return output

    def decode_word(w):
      indices = w.squeeze().tolist()
      return ''.join([idx_to_ch[idx] for idx in indices if idx != ch_to_idx['.']])

    def decode_tag(t):
      idx = t.squeeze().tolist()
      return idx_to_tg[idx]
    # ========= Instantiate the model 
    input_dim = len(ch_to_idx) + len(tg_to_idx)
    output_dim = len(ch_to_idx)
    model = BiLSTM(input_dim, hidden_dim, output_dim, n_layers)
    # ========= Train  
    if len(train_dataloader) > 0:
      optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
      train_loss_records = []
      dev_loss_records = []
      model.to(device)
      time_start = time.time()
      for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for src_form, src_tag, trgt_tag, trgt_form in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(src_form.to(device), src_tag.to(device), trgt_tag.to(device))
            # Compute loss
            loss = criterion(outputs.view(-1, output_dim), trgt_form.view(-1).to(device))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_loss_records.append(train_loss)
        # ========= Eval on dev
        if len(dev_dataloader) > 0:
          model.eval()
          dev_loss = 0.0
          with torch.no_grad():
              for src_form, src_tag, trgt_tag, trgt_form in dev_dataloader:
                  outputs = model(src_form.to(device), src_tag.to(device), trgt_tag.to(device))
                  loss = criterion(outputs.view(-1, output_dim), trgt_form.view(-1).to(device))  # Compute loss
                  dev_loss += loss.item()
          dev_loss /= len(dev_dataloader)
          dev_loss_records.append(dev_loss)
          if epoch % 5 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}')
      # ========= Finishing training      
      time_end = time.time()
      print(f'Training time: {time_end - time_start :.2f} seconds')
      # ========= Eval on Test
      model.eval()
      # heat maps needed: 
      # 1. accuracy
      # 2. model loss
      # 3. predictivity
      with open(f'{path}.1.prd', 'a', encoding='UTF8') as results:
        with torch.no_grad():
          counter = 0
          table_loss = {}
          table_acc = {}
          for src_form, src_tag, trgt_tag, trgt_form in test_dataloader:
            output = model(src_form.to(device), src_tag.to(device), trgt_tag.to(device))
            for i in range(output.shape[0]): # items in batch
              loss = criterion(output[i].view(-1, output_dim), trgt_form[i].view(-1).to(device))
              prediction = decode_word(output[i].argmax(dim=1))
              s_form, s_tag, t_tag, t_form = data_test[counter]
              
              result = 1 if prediction == t_form else 0
              if t_tag not in table_acc:
                table_acc[t_tag] = []
              table_acc[t_tag].append(result)
              
              if t_tag not in table_loss:
                table_loss[t_tag] = []
              table_loss[t_tag].append(loss.item())
              
              results.write(f'{result}\t{s_form}\t{s_tag}\t{t_tag}\t{t_form}\t{prediction}\t{loss.item()}\n')
              counter += 1
          for key, value_list in table_acc.items():
            avg = (sum(value_list) / len(value_list))
            heatmap_acc[encode_tag(tag)][encode_tag(key)] = avg
          for key, value_list in table_loss.items():
            avg = (sum(value_list) / len(value_list))
            heatmap_conf[encode_tag(tag)][encode_tag(key)] = avg
      print(f'Evaluated!')
  labels = list(idx_to_tg.values())[:-1]
  #==================== print heatmap ====================
  # uncomment the following lines to export the heatmap:
  # f_size = 4 if len(labels) > 6 else 10
  # # cmap = sns.cm.rocket_r
  # ax = sns.heatmap(heatmap_acc[:-1, :-1], annot=False, xticklabels=labels, yticklabels=labels, cmap='PiYG', cbar_kws={'label': 'Values'}, annot_kws={"size": f_size})
  # ax.set(xlabel='Target', ylabel='Source')
  # c_bar = ax.collections[0].colorbar
  # c_bar.set_label('Accuracy')
  # plt.xticks(fontsize=f_size)
  # plt.yticks(fontsize=f_size)
  # plt.savefig(f'{path}_heatmap_acc.png', dpi=300, bbox_inches='tight')
  #==================== write predictive weights ====================
  a =list(np.nanmean(heatmap_acc[:-1, :-1], axis=1)) # average (not considering NaN)
  m = min(a)
  b =[]
  for i in range(len(a)):
    w = a[i]
    if np.isnan(a[i]):
      w = m
    b.append([labels[i], w])
  b = sorted(b, key=lambda x: x[1], reverse=True)
  weight_dict = {}
  with open(f'{path}.predictive_power.tsv', 'w', encoding='UTF8') as results:
    for i in range(len(b)):
      w = b[i][1] 
      t = b[i][0]
      weight_dict[t] = w
      results.write(f'{b[i][1]:.2f}\t{b[i][0]}\n')
  #==================== weighted random sampling ====================
  predictive_weights = []
  for line in _pool:
    lemma, tags = line.strip().split('\t')
    if tags in weight_dict:
      predictive_weights.append(weight_dict[tags])
    else:
      predictive_weights.append(0.0)
  s = sum(predictive_weights)
  normalized_weights = [wg / s for wg in predictive_weights]    
  # Perform weighted random selection
  np.random.seed(100)
  samples = np.random.choice(_pool, p=normalized_weights, size=int(num_samples), replace=False)

  # =======    
  with open(f'{path}.oracle', 'a', encoding='utf-8') as w:
    for line in samples:
      # Remove selected instances from the original list
      _pool.remove(line)
      # getting the forms from oracle
      lemma, tags = line.strip().split('\t')
      form = get_form(lemma, tags)
      w.write(f'{lemma}\t{form}\t{tags}\n')
  # Write the remaining instances back to the pool file
  with open(f'{path}.pool', 'w', encoding='utf-8') as w:
    w.writelines(_pool)
 
  # splitting the oracle data into train and dev
  with open(f'{path}.oracle', 'r', encoding='utf-8') as f:
    oracle_data = f.readlines()
    
  # split random data into train, dev, test
  split_point = int(len(oracle_data) *  0.9)
  # Write train and dev data to separate files
  with open(f'{path}.2.trn', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[:split_point])
  with open(f'{path}.2.dev', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[split_point:])
  # the pool is temporarily used as test
  with open(f'{path}.2.tst', 'w', encoding='utf-8') as w:
      for line in _pool:
        lemma, tags = line.strip().split('\t')
        w.write(f'{lemma}\t{lemma}\t{tags}\n')


if __name__ == "__main__":
  main(*sys.argv[1:])
