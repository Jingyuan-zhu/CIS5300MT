"""Dictionary-based machine translation baseline using an explicit lexicon."""

from __future__ import annotations

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
import xml.etree.ElementTree as ET

import pandas as pd

from src.utils.data import load_translation_parquet


TokenDictionary = Mapping[str, str]


@dataclass
class DictionaryConfig:
    lowercase: bool = True
    strip_punctuation: bool = False
    unk_token: str = ""
    allow_identity_fallback: bool = True


class DictionaryTranslator:
    """Token-level replacement translator."""

    def __init__(self, lexicon: TokenDictionary, config: Optional[DictionaryConfig] = None) -> None:
        self.lexicon = dict(lexicon)
        self.config = config or DictionaryConfig()

    def translate(self, sentence: str) -> str:
        tokens = tokenize(sentence, lowercase=self.config.lowercase)
        translated = [self._translate_token(tok) for tok in tokens]
        return detokenize(translated)

    def _translate_token(self, token: str) -> str:
        if self.config.strip_punctuation and token in string.punctuation:
            return ""

        if token in self.lexicon:
            return self.lexicon[token]

        if not self.config.lowercase and token.lower() in self.lexicon:
            return self.lexicon[token.lower()]

        if self.config.allow_identity_fallback:
            return token

        return self.config.unk_token


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    pattern = r"\w+|[^\w\s]"
    tokens = re.findall(pattern, text, flags=re.UNICODE)
    return [token.lower() if lowercase else token for token in tokens]


def detokenize(tokens: Sequence[str]) -> str:
    sentence = ""
    for token in tokens:
        if not token:
            continue
        if not sentence:
            sentence = token
            continue
        if token in _NO_SPACE_BEFORE or (sentence and sentence[-1] in _NO_SPACE_AFTER):
            sentence += token
        else:
            sentence += " " + token
    return sentence.strip()


_NO_SPACE_BEFORE = {".", ",", ":", ";", "?", "!", "%", ")", "]", "}", "»"}
_NO_SPACE_AFTER = {"¿", "¡", "(", "[", "{", "«"}


def load_dictionary(dictionary_path: Path) -> TokenDictionary:
    if dictionary_path.suffix.lower() == ".json":
        with dictionary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items()}
        raise ValueError("JSON dictionary must be an object mapping strings to strings.")

    if dictionary_path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if dictionary_path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(dictionary_path, sep=sep, header=None, names=["source", "target"])
        return {str(row["source"]).strip(): str(row["target"]).strip() for _, row in df.iterrows()}

    if dictionary_path.suffix.lower() == ".xml":
        return _load_dictionary_from_xml(dictionary_path)

    raise ValueError(f"Unsupported dictionary format: {dictionary_path.suffix}")


def run_dictionary_baseline(args: argparse.Namespace) -> None:
    input_df = load_translation_parquet(
        args.input,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )

    if not args.dictionary:
        raise ValueError("A dictionary path must be provided via --dictionary for the baseline to run.")

    lexicon = load_dictionary(Path(args.dictionary))

    if not args.case_sensitive:
        lexicon = {key.lower(): value for key, value in lexicon.items()}

    translator = DictionaryTranslator(
        lexicon,
        DictionaryConfig(
            lowercase=not args.case_sensitive,
            strip_punctuation=args.strip_punctuation,
            unk_token=args.unk_token,
            allow_identity_fallback=not args.no_identity_fallback,
        ),
    )

    source_sentences = input_df["source"].astype(str).tolist()
    predictions = [translator.translate(sentence) for sentence in source_sentences]

    output_df = pd.DataFrame(
        {
            "source": source_sentences,
            "prediction": predictions,
        }
    )

    if "target" in input_df.columns:
        output_df["target"] = input_df["target"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_output(output_df, output_path)


def _write_output(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        df.to_csv(path, index=False, sep=sep)
    elif suffix in {".json", ".jsonl"}:
        orient = "records" if suffix == ".json" else "records"
        lines = suffix == ".jsonl"
        df.to_json(path, orient=orient, force_ascii=False, lines=lines)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def _load_dictionary_from_xml(path: Path) -> Dict[str, str]:
    tree = ET.parse(path)
    root = tree.getroot()
    lexicon: Dict[str, str] = {}
    for word_entry in root.findall(".//w"):
        source = word_entry.findtext("c")
        target = word_entry.findtext("d")
        if not source or not target:
            continue
        lexicon[source.strip()] = _clean_dictionary_target(target)
    return lexicon


_BRACE_PATTERN = re.compile(r"\{[^}]*\}")


def _clean_dictionary_target(text: str) -> str:
    cleaned = _BRACE_PATTERN.sub("", text)
    cleaned = cleaned.split(",")[0]
    cleaned = cleaned.split(";")[0]
    cleaned = cleaned.split(" (")[0]
    return cleaned.strip()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dictionary-based translation baseline.")
    parser.add_argument("--input", required=True, help="Path to the evaluation dataset parquet file.")
    parser.add_argument("--output", required=True, help="Where to store the generated translations.")
    parser.add_argument("--dictionary", required=True, help="Path to a bilingual dictionary (json/csv/tsv/xml).")
    parser.add_argument("--source-lang", default="en", help="Source language key.")
    parser.add_argument("--target-lang", default="es", help="Target language key.")
    parser.add_argument("--unk-token", default="", help="Token to use when the dictionary cannot translate a word.")
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Remove punctuation tokens from the output.",
    )
    parser.add_argument(
        "--no-identity-fallback",
        action="store_true",
        help="Disable copying untranslated tokens directly to the output.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Keep dictionary lookups case-sensitive (default: insensitive).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dictionary_baseline(args)


if __name__ == "__main__":
    main()

