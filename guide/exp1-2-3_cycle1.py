import sys
import os
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
  trn_dev_split_ratio = 0.9  
  #==================== startup ====================
  data_file = f'{data_path}/{lang}.tsv'
  global working_dir
  working_dir = f'{work_path}' 
  if not os.path.exists(working_dir):
    os.makedirs(working_dir)
  path = f'{working_dir}/{lang}' 
  load_oracle_knowledge(data_file)
  #==================== Random sampling ====================  
  # read the original data file
  pool = []
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
      lemma, form, tags = line.strip().split('\t')
      pool.append(f'{lemma}\t{tags}\n')
  # Randomly select instances
  random.seed(100)
  samples = random.sample(pool, int(num_samples))
  oracle_data = []
  with open(f'{path}.oracle', 'w', encoding='utf-8') as w:
    for line in samples:
      # Remove selected instances from the original list
      pool.remove(line)
      # getting the forms from oracle
      lemma, tags = line.strip().split('\t')
      form = get_form(lemma, tags)
      w.write(f'{lemma}\t{form}\t{tags}\n')
  # Write the remaining instances back to the pool file
  with open(f'{path}.pool', 'w', encoding='utf-8') as f:
    f.writelines(pool)
    
  #==================== write train, dev, and test files
  
  # splitting the oracle data into train and dev
  with open(f'{path}.oracle', 'r', encoding='utf-8') as f:
    oracle_data = f.readlines()
  # Shuffle and split the data
  random.shuffle(oracle_data)
  split_point = int(len(oracle_data) *  trn_dev_split_ratio)
  # Write train and dev data to separate files
  with open(f'{path}.1.trn', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[:split_point])
  with open(f'{path}.1.dev', 'w', encoding='utf-8') as w:
    w.writelines(oracle_data[split_point:])
  # the pool is temporarily used as test
  with open(f'{path}.1.tst', 'w', encoding='utf-8') as w:
    with open(f'{path}.pool', 'r', encoding='utf-8') as f:
      for line in f:
        lemma, tags = line.strip().split('\t')
        w.write(f'{lemma}\t{lemma}\t{tags}\n')
  
if __name__ == "__main__":
  main(*sys.argv[1:])
