#!/bin/bash
arch=${2:-tagtransformer}

lr=0.001
scheduler=warmupinvsqr
max_steps=500
warmup=500
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${3:-0.3}
ckpt_dir=checkpoints
#
data_path=data

num_samples=400 # 100 for mwf
num_most_confident=300 #70 for mwf

for lang in eng ckb lat rus tur pbs khk; do
  for exp in 1 2 3; do
    work_path=results/$lang.exp$exp    
    for cycle in {1..5}; do
      if [ $cycle -eq 1 ]; then
        # cycle 1
        python guide/exp1-2-3_cycle1.py "$lang" "$data_path" "$work_path" "$num_samples"
      else
        # cycle 2-5
        if [ $exp -le 2 ]; then
          python guide/exp1-2_cycle2-5.py "$lang" "$data_path" "$work_path" "$cycle" "$exp" "$num_samples"
        else
          python guide/exp3_cycle2-5.py "$lang" "$data_path" "$work_path" "$cycle" "$num_samples" "$num_most_confident"
        fi
      fi
      python src/train.py \
        --dataset sigmorphon17task1 \
        --train $work_path/$lang.$cycle.trn \
        --dev $work_path/$lang.$cycle.dev \
        --test $work_path/$lang.$cycle.tst \
        --model $ckpt_dir/$lang.$exp/$cycle \
        --decode greedy --max_decode_len 32 \
        --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
        --label_smooth $label_smooth --total_eval $total_eval \
        --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
        --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
        --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
      transformer_results=$ckpt_dir/$lang.$exp/$cycle.decode.test.tsv
      python guide/convert_transformer_results.py "$lang" "$work_path" "$cycle" "$transformer_results"
    done
  done
  # ==============================
  exp=4
  work_path=results/$lang.exp$exp

  for cycle in {2..5}; do
    if [ $cycle -eq 2 ]; then
      # cycle 1 and 2
      python guide/exp4_cycle1-2.py "$lang" "$data_path" "$work_path" "$num_samples"
    else
      # cycle 3-5
      python guide/exp4_cycle3-5.py "$lang" "$data_path" "$work_path" "$last_predictions" "$cycle" "$num_samples"
    fi    
    python src/train.py \
      --dataset sigmorphon17task1 \
      --train $work_path/$lang.$cycle.trn \
      --dev $work_path/$lang.$cycle.dev \
      --test $work_path/$lang.$cycle.tst \
      --model $ckpt_dir/$lang.$exp/$cycle \
      --decode greedy --max_decode_len 32 \
      --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
      --label_smooth $label_smooth --total_eval $total_eval \
      --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
      --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
      --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
      transformer_results=$ckpt_dir/$lang.$exp/$cycle.decode.test.tsv
      python guide/convert_transformer_results.py "$lang" "$work_path" "$cycle" "$transformer_results"
  done
done
