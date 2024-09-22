#==================== load oracle data ====================
oracle_knowledge = {}
unique_tags = set()

def load_oracle_knowledge(data_file):
  global oracle_knowledge
  with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
      lemma, form, tags = line.strip().split('\t')
      unique_tags.add(tags)
      k = f'{lemma}@{tags}'
      oracle_knowledge[k] = form
     
#==================== final report ====================
def main(lang, data_path, working_path, table_path):
  data_file = f'{data_path}/{lang}.tsv'
  load_oracle_knowledge(data_file)
    
  for exp in range(1, 5):
    working_dir = f'{working_path}/{lang}.exp{exp}'
    # Final stat of oracle requests
    seen_data = {}
    S_N = 0 # Asked_without_prediction
    S_W = 0 # Asked_with_wrong_prediction
    S_C = 0 # Asked_with_correct_prediction
    with open(f'{working_dir}/oracle_stat.tsv', 'r', encoding='utf-8') as f:
      for line in f:
        result, predicted, form, lemma, tags = line.strip().split('\t')
        k = f'{lemma}@{tags}'
        if k in seen_data:
          print(f'Duplicate: {k}')
        else:
          seen_data[k] = result
        if result == '1':
          S_C += 1
        elif result == '-1':
          S_W += 1
        elif result == '0':
          S_N += 1  
    with open(f'{table_path}/report_oracle_stat.txt', 'a', encoding='utf-8') as w:
      w.write(f'{lang}\tExp.{exp}\t{S_N + S_W + S_C}\t{S_N}\t{S_W}\t{S_C}\n')
  
    # Accuracy in each cycle
    with open(f'{table_path}/report_acc_cycles.txt', 'a', encoding='utf-8') as acc_report:
      for cycle in range(1, 6):
        unseen_data = {}
        P_W = 0
        P_C = 0
        if exp == 4 and cycle == 1:
          continue
        with open(f'{working_dir}/{lang}.{cycle}.prd', 'r', encoding='utf-8') as f:
          for line in f:
            line = line.strip()
            if line:
              lemma, tags, prediction, loss = line.split('\t')
              k = f'{lemma}@{tags}'
              if k not in seen_data:
                real_form = oracle_knowledge[k]
                is_true = '?'
                if real_form == prediction:
                  if k not in unseen_data:
                    is_true = '1'
                    P_C += 1
                    unseen_data[k] = [prediction, 1]
                else:
                  if k not in unseen_data:
                    is_true = '0'
                    P_W += 1
                    unseen_data[k] = [prediction, -1]
          acc = 100 * (P_C) / (P_W + P_C)
          acc_report.write(f'{lang}\tExp.{exp}\t{cycle}\t{acc :.2f}\n')
          # Final accuracy and NES
          count_all_forms = len(oracle_knowledge.items())
          NES = (1 - (P_W + S_N + S_W) / count_all_forms) * 100
          if cycle == 5:
            with open(f'{table_path}/report_final_acc_NES.txt', 'a', encoding='utf-8') as final_report:
              final_report.write(f'{lang}\tExp.{exp}\t{acc :.2f}\t{NES :.2f}\n')


table_path = 'tables'
with open(f'{table_path}/report_final_acc_NES.txt', 'w', encoding='utf-8') as w:
  w.write(f'lang\tExp\tAcc\tNES\n')
with open(f'{table_path}/report_oracle_stat.txt', 'w', encoding='utf-8') as w:
  w.write(f'lang\tExp\tAll\tN\tW\tC\n')
with open(f'{table_path}/report_acc_cycles.txt', 'w', encoding='utf-8') as w:
  w.write(f'lang\tExp\tcycle\tAcc\n')
  
  
langs = ['tur', 'ckb', 'eng', 'khk', 'rus', 'lat',  'pbs', 'mwf']
for lang in ['tur', 'ckb', 'eng', 'khk', 'rus', 'lat', 'pbs']:
  main(lang, data_path='data', working_path='results', table_path=table_path)