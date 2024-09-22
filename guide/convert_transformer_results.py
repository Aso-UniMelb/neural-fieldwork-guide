import sys

def main(lang, working_dir, cycle, transformer_results):
  # convert transformer predictions of previous cycle to our prd format
  path = f'{working_dir}/{lang}'
  with open(f'{path}.{cycle}.prd', 'w', encoding='utf-8') as w:
    temp = []
    with open(f'{path}.{cycle}.tst', 'r', encoding='utf-8') as f:
      for line in f:
        l = line.strip().split('\t')
        temp.append([l[1], l[2]])
    with open(transformer_results, 'r', encoding='utf-8') as f:
      next(f) # skip header
      counter = 0
      for line in f:
        l = line.strip().split('\t')
        if len(l) == 4:
          prediction, lemma, loss, dist = line.strip().split('\t')
          prediction = prediction.replace(' ', '')
          tags = temp[counter][1]
          lemma = temp[counter][0]
          w.write(f'{lemma}\t{tags}\t{prediction}\t{loss}\n')
        counter += 1

if __name__ == "__main__":
  main(*sys.argv[1:])