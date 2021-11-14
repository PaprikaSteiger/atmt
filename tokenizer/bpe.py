from pathlib import Path
import json
import typing as t

import tokenizers.decoders
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder

from seq2seq.data.dictionary import Dictionary
from preprocess import make_binary_dataset


def train_bpe_tokenizer(inputfile_trt: t.List[str], inputfile_src: t.List[str], outputpath: Path, vocab_size=5000):
    tokenizer = Tokenizer(BPE(end_of_word_suffix="<w>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, end_of_word_suffix="<w>", special_tokens=["<pad>"])
    tokenizer.train(inputfile_trt + inputfile_src, trainer)
    tokenizer.model.save(str(outputpath.parent))
    tokenizer.save(str(outputpath))
    return tokenizer


def load_bpe_tokenizer(tokenizer_file: str):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    tokenizer.decoder = tokenizers.decoders.BPEDecoder(suffix="<w>")
    return tokenizer


def load_bpe_model(vocab_file:str, merge_file: str):
    return BPE.from_file(vocab=vocab_file, merges=merge_file, end_of_word_suffix="<w>")


def create_dict(voca_json: str, dict_path: Path, file_prefix: str = ""):
    dict = json.load(open(voca_json, "r", encoding="utf8"))
    outfiles = []
    for ending in ["fr", "en"]:
        outfile = dict_path / f"{file_prefix}dict.{ending}"
        outfiles.append(outfile)
        with open(outfile, "w", encoding="utf8") as out:
            for k, v in dict.items():
                out.write(f"{k} {v}\n")
    return outfiles


def decode_file(infile:str, outfile: str, suffix: str="<w>"):
    decoder = BPEDecoder(suffix=suffix)
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
    # # paths to data
    # tokenizer_dir = Path(__file__).parent
    # prepared_path = (tokenizer_dir.parent / "data" / "en-fr" / "prepared")
    # preprocessed_path = (tokenizer_dir.parent / "data" / "en-fr" / "preprocessed")
    # # tiny dataset
    # tiny_t= train_bpe_tokenizer(
    #     inputfile_trt=[str( preprocessed_path /"tiny_train.en")],
    #     inputfile_src=[str(preprocessed_path / "tiny_train.fr")],
    #     outputpath=tokenizer_dir / "tiny" / "tiny_tokenizer-json",
    #     vocab_size=500
    # )
    # #t = load_bpe_tokenizer(tokenizer_file=str(tokenizer_dir / "tokenizer.json"))
    # tiny_outfiles = create_dict(voca_json=str(tokenizer_dir / "tiny" / "vocab.json"), dict_path=prepared_path, file_prefix="bpe_tiny_")
    # dictionary_tiny = Dictionary.load(tiny_outfiles[0])
    # endings = ["fr", "en"]
    # for ending in endings:
    #     inn = str(preprocessed_path / f"tiny_train.{ending}")
    #     out = input_file=str(preprocessed_path / f"bpe_tiny_train.{ending}")
    #     tokenize_bpe(tokenizer=tiny_t, infile=inn, outfile=out)
    #     make_binary_dataset(
    #         input_file=str(preprocessed_path / f"bpe_tiny_train.{ending}"),
    #         output_file=str(prepared_path / f"bpe_tiny_train.{ending}"),
    #         dictionary=dictionary_tiny
    #     )
    # # normal data
    # t = train_bpe_tokenizer(
    #     inputfile_trt=[str(preprocessed_path / "train.en"), str(preprocessed_path / "valid.en")],
    #     inputfile_src=[str(preprocessed_path / "train.fr"), str(preprocessed_path / "valid.fr")],
    #     outputpath=tokenizer_dir / "normal" / "tokenizer-json",
    #     vocab_size=3997
    # )
    # outfiles = create_dict(voca_json=str(tokenizer_dir / "normal" / "vocab.json"), dict_path=prepared_path, file_prefix="bpe_")
    # dictionary = Dictionary.load(outfiles[0])
    # files = ["train", "test", "valid"]
    # for ending in endings:
    #     for file in files:
    #         inn = str(preprocessed_path / f"{file}.{ending}")
    #         out = str(preprocessed_path / f"bpe_{file}.{ending}")
    #         tokenize_bpe(tokenizer=t, infile=inn, outfile=out)
    #         make_binary_dataset(
    #             input_file=str(preprocessed_path / f"bpe_{file}.{ending}"),
    #             output_file=str(prepared_path / f"bpe_{file}.{ending}"),
    #             dictionary=dictionary
    #         )

    decode_file(
        infile=Path(__file__).parent.parent / "deep_bpe_continued" /"output.txt",
        outfile=Path(__file__).parent.parent / "deep_bpe_continued" /"decoded_output.txt",
    )

