import sys
import numpy as np
import random

#==================== load oracle data ====================
oracle_knowledge = {}
unique_tags = set()
working_dir = ''

def load_oracle_knowledge(data_file):
  global oracle_knowledge
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
      lemma, form, tags = line.strip().split('\t')
      unique_tags.add(tags)
      k = f'{lemma}@{tags}'
      oracle_knowledge[k] = form
  
# get the form from oracle
def get_form(lemma, tags, predicted):
  k = f'{lemma}@{tags}'
  if k in oracle_knowledge:
    form = oracle_knowledge[k]
    with open(f'{working_dir}/oracle_stat.tsv', 'a', encoding='utf-8') as w:
      if(predicted == ''):
        result = 0
      else:
        result = 1 if form == predicted else -1
      w.write(f'{result}\t{predicted}\t{form}\t{lemma}\t{tags}\n')
    return form
  else:
      return ''

def main(lang, data_path, work_path, last_predictions, cylce_count, num_samples):
  #==================== options ====================
  trn_dev_split_ratio = 0.9
  #==================== startup ====================
  global working_dir
  working_dir = f'{work_path}'
  path = f'{working_dir}/{lang}'
  data_file = f'{data_path}/{lang}.tsv'
  load_oracle_knowledge(data_file)
  cylce_count = int(cylce_count)
  
  weight_dict = {}
  with open(f'{path}.predictive_power.tsv', 'r', encoding='UTF8') as f:
    for line in f:
      w, tags = line.strip().split('\t')
      weight_dict[tags] = float(w)
  min_weight = min(weight_dict.values())
  
  # convert transformer predictions of previous cycle to our prd format
  losses = []
  test_file = []
  predictive_weights = []
  with open(f'{path}.{cylce_count - 1}.prd', 'r', encoding='utf-8') as w:
    for line in w:
      l = line.strip().split('\t')
      if len(l) == 4:
        lemma, tags, prediction, loss = l
        test_file.append(f'{lemma}\t{tags}\t{prediction}\t{loss}\n')
        losses.append(float(loss))
        if tags in weight_dict:
          predictive_weights.append(weight_dict[tags])
        else:
          predictive_weights.append(min_weight)
  average_loss = sum(losses) / len(losses)
  #==================== weighted random sampling ====================  
  s = sum(predictive_weights)
  normalized_weights = [w / s for w in predictive_weights]    
  # Perform weighted random selection  
  random.seed(100)
  samples = np.random.choice(test_file, p=normalized_weights, size=int(num_samples), replace=False)
  new_from_oracle = []
  # getting data from oracle
  with open(f'{path}.oracle', 'a', encoding='utf-8') as w:
    for line in samples:
      lemma, tags, predicted, loss = line.strip().split('\t')
      prediction = ''
      if float(loss) < average_loss:
        prediction = predicted
      form = get_form(lemma, tags, prediction)
      new_from_oracle.append(f'{lemma}\t{tags}\n')
      w.write(f'{lemma}\t{form}\t{tags}\n')
  write_trn_dev_tst(path, cylce_count, trn_dev_split_ratio)
  
  
def write_trn_dev_tst(path, cylce_count, trn_dev_split_ratio):
  with open(f'{path}.oracle', 'r', encoding='utf-8') as f:
    oracle_data = f.readlines()  
  random.seed(100)
  random.shuffle(oracle_data)
  split_point = int(len(oracle_data) *  trn_dev_split_ratio)
  # Write train and dev data to separate files
  with open(f'{path}.{cylce_count}.trn', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[:split_point])
  with open(f'{path}.{cylce_count}.dev', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[split_point:])  
      
  # remove the recent oracle data from the pool
  with open(f'{path}.pool', 'r', encoding='utf-8') as f:
    old_pool = f.readlines()
  oracle_items = []
  for line in oracle_data:
    lemma, form, tags = line.strip().split('\t')
    oracle_items.append(f'{lemma}\t{tags}\n')
  updated_pool = []
  with open(f'{path}.pool', 'w', encoding='utf-8') as w:
    for line in old_pool:
      if line not in oracle_items:
        updated_pool.append(line)
        w.write(line)
  # updated pool is used as test set
  with open(f'{path}.{cylce_count}.tst', 'w', encoding='utf-8') as w:
    for line in updated_pool:
      lemma, tags = line.strip().split('\t')
      w.write(f'{lemma}\t{lemma}\t{tags}\n')
        

if __name__ == "__main__":
  main(*sys.argv[1:])
