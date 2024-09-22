import matplotlib.pyplot as plt

data = {}
with open('tables/report_acc_cycles.txt', 'r', encoding='utf-8') as f:  
  next(f) #skip header
  for line in f:
    lang,	exp, cycle,	acc = line.strip().split('\t')
    if lang not in data:
      data[lang] = {}
    if exp not in data[lang]:
      data[lang][exp] = {}
    data[lang][exp][int(cycle)] = float(acc)

langs = list(data.keys())
count = 0
plt.figure(figsize=(15, 4))
for lang in langs:
  plt.subplot(1, 8, count + 1)
  plt.axis([0.5, 5.5, 20, 100])
  plt.xticks(range(1, 6))
  for exp, values in data[lang].items():
    plt.plot(list(values.keys()), list(values.values()), marker='o', label=exp)
  plt.xlabel('Cycle')
  if count == 0:
    plt.ylabel('Accuracy')
    plt.legend(title='Experiments')  
  title = langs[count]
  plt.title(title)
  count += 1  
  plt.tight_layout()

plt.savefig(f'figures/acc_cycles.png', dpi=300, bbox_inches='tight')
