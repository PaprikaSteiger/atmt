##bpe-continued
pretrained model loaded and continued training with bpe\
### bpe model

### nmt model

--data /home/melvin/Desktop/mt-assignments/atmt/data/en-fr/prepared\
--source-lang fr\
--target-lang en\
--batch-size 128\
--max-epoch 500\
--restore-file /home/melvin/Desktop/mt-assignments/atmt/assignments/03/baseline/checkpoints/checkpoint_best.pt \
--data-prefix bpe_ \
--save-dir "bpe_continued"


### bleu score
sacrebleu /home/melvin/Desktop/mt-assignments/atmt/data/en-fr/raw/test.en -i /home/melvin/Desktop/mt-assignments/atmt/bpe_continued/postprocesed_decoded_output.txt -m bleu -w 4
{
 "name": "BLEU",
 "score": 18.8224,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "51.1/24.5/13.6/7.4 (BP = 1.000 ratio = 1.045 hyp_len = 4066 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}



# ignore this for now
## pbe-from-scratch
pretrained model not loaded, trained from scratch with bpe
### bpe model
same as above
### nmt model
--data /home/melvin/Desktop/mt-assignments/atmt/data/en-fr/prepared \
--source-lang fr\
--target-lang en \
--batch-size 128 \
--max-epoch 500 \
--data-prefix bpe_ \
--save-dir "bpe_from_scratch"

### translation settings
--data /home/melvin/Desktop/mt-assignments/atmt/data/en-fr/prepared \
--dicts /home/melvin/Desktop/mt-assignments/atmt/data/en-fr/prepared \
--checkpoint-path /home/melvin/Desktop/mt-assignments/atmt/bpe_from_scratch/checkpoint_best.pt \
--output /home/melvin/Desktop/mt-assignments/atmt/bpe_from_scratch/output.txt \
--data-prefix bpe_ \
--max-len 10


