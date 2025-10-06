
import pandas as pd
import argparse
import pathlib
import re

APOS = ("'", "’", "ʼ", "ʹ", "‛", "＇")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def count_occurrences(text: str, expressions: set[str]) -> int:
    if not text:
        return 0
    t = text.lower()
    c =0
    for expr in expressions:
        c += t.count(expr)
    return c

def count_all_caps(text: str) -> int:
    tokens = text.split()
    punct = ".,!?;:\"'()[]<>"
    n = 0
    for tok in tokens:
        t = tok.strip(punct)
        if len(t)>0 and t.isupper(): n += 1
    return n

def has_repeated_word(text: str) -> int:
    text = str(text)
    punct = ".,!?;:\"'()[]<>"
    prev = None
    for tok in text.split():
        t = tok.strip(punct)
        if not t:
            continue
        tnorm = t.casefold()
        if prev is not None and tnorm == prev:
            return 1
        prev = tnorm
    return 0

def count_tokens_starting_with(text: str, prefix: str ) -> int:
    n = 0
    tokens = text.split()
    punct = ".,!?;:\"'()[]<>"
    for tok in tokens:
        t = tok.strip(punct)
        if  t.startswith(prefix):
            n += 1
    return n


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = "".join("'" if ch in APOS else ch for ch in text)
    return [t.casefold() for t in TOKEN_RE.findall(text)]


def count_polar_expressions(text: str,
                            pos_words: set[str],
                            neg_words: set[str],
                            intensifiers: set[str],
                            negations: set[str],
                            pos_emojis: set[str],
                            neg_emojis: set[str]) -> tuple[int, int]:
    if not text:
        return 0, 0
    toks = tokenize(text)
    L = len(toks)
    pos = 0
    neg = 0
    joined = text.casefold()
    pe = sum(joined.count(e) for e in pos_emojis)
    ne = sum(joined.count(e) for e in neg_emojis)
    i = 0
    while i < L:
        w = toks[i]
        p = w in pos_words
        n = w in neg_words
        if p or n:
            j = i - 1
            flip = False
            while j >= 0 and (toks[j] in intensifiers or toks[j] in negations):
                if toks[j] in negations:
                    flip = not flip
                j -= 1
            if p:
                if flip:
                    neg += 1
                else:
                    pos += 1
            else:
                if flip:
                    pos += 1
                else:
                    neg += 1
        i += 1
    return pos + pe, neg + ne


def load_lexiques() -> dict[str,set[str]]:
    lexique_folder = "datasets/lexiques"
    folder = pathlib.Path(lexique_folder)


    lexiques: dict[str, set[str]] = {}
    for p in folder.iterdir():
        name = p.stem
        expressions: set[str] = []

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                expr = line.rstrip("\n\r")
                if expr != "":
                    expressions.append(expr.lower())
        lexiques[name] = expressions
        
    return lexiques


def add_features(df: pd.DataFrame, lexiques: dict[str, list[str]]) -> pd.DataFrame:
    emoticon_pos =lexiques["positives_emojis"]
    emoticon_neg = lexiques["negatives_emojis"]
    word_pos = lexiques["positives_words"]
    word_neg = lexiques["negatives_words"]
    intensifiers = lexiques["intensifiers"]
    negation_word = lexiques["negation_words"]

    print("positives_emojis:", len(emoticon_pos))
    print("negatives_emojis:", len(emoticon_neg))
    print("positives_words:", len(word_pos))
    print("negatives_words:", len(word_neg))
    print("intensifiers:", len(intensifiers))
    print("negations:", len(negation_word))

    current = df["SentimentText"].astype(str)
    df["nb_mots"] = current.apply( lambda x : len(str(x).split()))
    df["nb_caracteres"]  = current.apply(lambda x: len(str(x)))
    df["nbr_emoticones_positifs"] = current.apply(lambda x: count_occurrences(x, emoticon_pos))
    df["nbr_emoticones_negatifs"] = current.apply(lambda x: count_occurrences(x, emoticon_neg))
    df["nb_mots_pos"] = current.apply(lambda x: count_occurrences(x, word_pos))
    df["nb_mots_neg"] = current.apply(lambda x: count_occurrences(x, word_neg))
    df["nb_intensifieurs"] = current.apply(lambda x: count_occurrences(x, intensifiers))
    df["nb_mots_neg"] = current.apply(lambda x: count_occurrences(x, negation_word))
    df["nb_majuscules"] = current.apply(lambda x: count_all_caps(x))
    df["nb_mentions"] = current.apply(lambda x: count_tokens_starting_with(x, '#'))
    df["nb_hashtags"] = current.apply(lambda x: count_tokens_starting_with(x, '@'))
    df["nb_exclamations"] = current.apply(lambda x: str(x).count("!"))
    df["nb_questions"]    = current.apply(lambda x: str(x).count("?"))
    df["mots_repetes"] = current.apply(lambda x: has_repeated_word(x))
    res = current.apply(lambda x: count_polar_expressions(x, word_pos, word_neg, intensifiers, negation_word, emoticon_pos, emoticon_neg))
    df["exp_pos"] = res.apply(lambda t: t[0])
    df["exp_neg"] = res.apply(lambda t: t[1])
    return df

def extract_features(train_data_file:str,saving_file:str):
    lexiques = load_lexiques()
    #data_loaded = load_data(train_data_file)
    for exp_type, exprs in lexiques.items():
        print(f"\n[lexique] {exp_type}")
        for expr in exprs :
            print(f" - {expr}")
    data_loaded = pd.read_csv(train_data_file,  encoding="utf-8",encoding_errors="replace",index_col="ItemID")
    tweet_features = add_features(data_loaded,lexiques)
    tweet_features.to_csv(saving_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from a given file and save them"
    )
    parser.add_argument("train_data", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    extract_features(args.train_data,args.output_file)

