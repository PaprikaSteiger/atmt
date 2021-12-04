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
translation setting: max length 70

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


## deeper model with pbe

### model settings
'batch_size': 128,
'encoder_num_layers': 3,
'decoder_num_layers': 3,
translation setting: max length 70
### bleu score
{
 "name": "BLEU",
 "score": 9.2782,
 "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.0.0",
 "verbose_score": "40.9/13.3/5.7/2.4 (BP = 1.000 ratio = 1.088 hyp_len = 4235 ref_len = 3892)",
 "nrefs": "1",
 "case": "mixed",
 "eff": "no",
 "tok": "13a",
 "smooth": "exp",
 "version": "2.0.0"
}



