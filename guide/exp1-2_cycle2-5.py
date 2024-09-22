import sys
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

def main(lang, data_path, work_path, cylce_count, exp, num_samples):
  #==================== options ====================
  trn_dev_split_ratio = 0.9
  #==================== startup ====================
  data_file = f'{data_path}/{lang}.tsv'
  global working_dir
  working_dir = f'{work_path}'
  path = f'{working_dir}/{lang}'
  load_oracle_knowledge(data_file)
  cylce_count = int(cylce_count)
  
  #==================== Sampling #1: Random without predictions  ====================
  if exp ==  '1':
    with open(f'{path}.pool', 'r', encoding='utf-8') as f:
      pool = f.readlines()
    random.seed(100)
    samples = random.sample(pool, int(num_samples))
    with open(f'{path}.oracle', 'a', encoding='utf-8') as w:
      for l in samples:
        lemma, tags = l.strip().split('\t')
        form = get_form(lemma, tags, '')
        w.write(f'{lemma}\t{form}\t{tags}\n')    
  #==================== Sampling #2: Random with predictions for confident samples ====================
  if exp ==  '2':
    losses = [] # collect losses for experimnet 2
    with open(f'{path}.{cylce_count - 1}.prd', 'r', encoding='utf-8') as w:
      for line in w:
        l = line.strip().split('\t')
        if len(l) == 4:
          lemma, tags, prediction, loss = l
          losses.append(float(loss))
    with open(f'{path}.pool', 'r', encoding='utf-8') as f:
      pool = f.readlines()
    random.seed(100)
    samples = random.sample(pool, int(num_samples))
    # read the predictions file if loss is less than the average loss send the prediction to oracle
    average_loss = sum(losses) / len(losses)
    with open(f'{path}.oracle', 'a', encoding='utf-8') as w:
      with open(f'{path}.{cylce_count - 1}.prd', 'r', encoding='utf-8') as f:
        for line in f:
          lemma, tags, predicted, loss = line.strip().split('\t')
          k = f'{lemma}\t{tags}\n'
          if k in samples:
            if float(loss) < average_loss:
              form = get_form(lemma, tags, predicted)
            else:
              form = get_form(lemma, tags, '')            
            w.write(f'{lemma}\t{form}\t{tags}\n')    
  #==================== 
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
