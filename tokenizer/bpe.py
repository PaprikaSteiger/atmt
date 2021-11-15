from pathlib import Path
import json
import typing as t
import argparse

import tokenizers.decoders
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder

from seq2seq.data.dictionary import Dictionary
from preprocess import make_binary_dataset


def train_bpe_tokenizer(inputfile_trt: t.List[str], inputfile_src: t.List[str], outputpath: Path, vocab_size=3997, suffix="<w>"):
    tokenizer = Tokenizer(BPE(end_of_word_suffix=suffix))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, end_of_word_suffix=suffix)
    tokenizer.train(inputfile_trt + inputfile_src, trainer)
    tokenizer.model.save(str(outputpath.parent))
    tokenizer.save(str(outputpath))
    return tokenizer


def load_bpe_tokenizer(tokenizer_file: str):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    return tokenizer


def load_bpe_model(vocab_file:str, merge_file: str, suffix="<w>"):
    return BPE.from_file(vocab=vocab_file, merges=merge_file, end_of_word_suffix=suffix)


def create_dict(voca_json: str, dict_paths: t.List[Path] , file_prefix: str = "bpe_"):
    dict = json.load(open(voca_json, "r", encoding="utf8"))
    outfiles = []
    for file in dict_paths:
        outfile = file.parent / f"{file_prefix}{file.name}"
        outfiles.append(outfile)
        with open(outfile, "w", encoding="utf8") as out:
            for k, v in dict.items():
                out.write(f"{k} {v}\n")
    return outfiles


def decode_file(infile:str, outfile: str, decoder: BPEDecoder):
    with open(infile, "r", encoding="utf8") as inn, open(outfile, "w", encoding="utf8") as out:
        for line in inn:
            line = decoder.decode(line.split())
            line = line.replace("& apos ; ", "&apos;")
            out.write(line + "\n")


def tokenize_bpe(tokenizer: tokenizers.Tokenizer, infile: str, outfile: str):
    with open(infile, "r", encoding="utf8") as inn, open(outfile, "w", encoding="utf8") as out:
        for line in inn:
            line = tokenizer.encode(line)
            out.write(" ".join(line.tokens) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir",
                        help="Path to directory were tokenizer model is stored and or should be created",
                        type=Path,
                        default=None,
                        )
    parser.add_argument("--data-dir",
                        help="Path to directory containing the preprocessed data files",
                        type=Path,
                        default=None,
                        )
    parser.add_argument("--output-dir",
                        help="Path to directory were prepared data files should be outputted",
                        type=Path,
                        default=None,
                        )
    parser.add_argument("--data-prefix",
                        help="Prefix attached to file name to marke bpe",
                        type=str,
                        default="bpe_"
                        )
    parser.add_argument("--preprocess",
                        action='store_true',
                        help="If activated, script preprocesses given data files and creates bpe files",
                        )
    parser.add_argument("--postprocess",
                        action='store_true',
                        help="If activated, script takes bpe encoded input and outputs decoded version",
                        )
    parser.add_argument("--postprocess-file",
                        help="File to be decoded with bpe model",
                        type=Path,
                        default=None,
                        )
    parser.add_argument("--bpe-suffix",
                        type=str,
                        help="Suffix that marks the end of a word in the bpe model",
                        default="<w>"
                        )
    parser.add_argument("--vocabsize",
                        type=int,
                        help="Size of vacabulary of bpe model",
                        default=3997
                        )
    # paths to data
    args = parser.parse_args()
    if args.tokenizer_dir:
        tokenizer_dir = args.tokenizer_dir
    else:
        tokenizer_dir = Path(__file__).parent
    if args.data_dir:
        preprocessed_path = args.data_dir
    else:
        preprocessed_path = (tokenizer_dir.parent / "data" / "en-fr" / "preprocessed")

    if args.output_dir:
        prepared_path = args.output_dir
    else:
        prepared_path = (tokenizer_dir.parent / "data" / "en-fr" / "prepared")
    # normal data
    if args.preprocess:
        tokenizer = None
        for file in tokenizer_dir.iterdir():
            if file.name == "tokenizer.json":
                tokenizer = load_bpe_tokenizer(str(file))
                break
        if not tokenizer:
            tokenizer = train_bpe_tokenizer(
                inputfile_trt=[str(preprocessed_path / "train.en")],
                inputfile_src=[str(preprocessed_path / "train.fr")],
                outputpath=tokenizer_dir / "tokenizer.json",
                vocab_size=args.vocabsize,
                suffix=args.bpe_suffix,
            )
        dict_files = create_dict(
            voca_json=(tokenizer_dir / "vocab.json"),
            dict_paths=[(prepared_path / "dict.en"), (prepared_path / "dict.fr")],
            file_prefix=args.data_prefix
        )
        dictionary = Dictionary.load(dict_files[0])
        for file in list(preprocessed_path.iterdir()):
            out = preprocessed_path / f"{args.data_prefix}{file.name}"
            tokenize_bpe(tokenizer=tokenizer, infile=file, outfile=out)
            out_prepared = prepared_path / out.name
            make_binary_dataset(
                input_file=out,
                output_file=out_prepared,
                dictionary=dictionary,
            )
    if args.postprocess:
        if args.postprocess_file:
            decoder = BPEDecoder(suffix=args.bpe_suffix)
            decode_file(
                infile=args.postprocess_file,
                outfile=args.postprocess_file.parent / f"decoded_{args.postprocess_file.name}.txt",
                decoder=decoder
            )
        else:
            raise ValueError("No file to decode was specified")
