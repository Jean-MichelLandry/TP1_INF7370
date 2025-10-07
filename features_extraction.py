
import pandas as pd
import argparse
import pathlib
import re
import html

APOS = ("'", "’", "ʼ", "ʹ", "‛", "＇")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)






def count_occurrences_emojis(text: str, expressions: set[str]) -> int:
    c = 0
    if not text or len(text)==0:
        return 0
        
    s = html.unescape(text).casefold()
    for expr in expressions:
        c+= s.count(expr.casefold())
    return c

def count_occurrences_words(text: str, expressions: set[str]) -> int:
    if not text or not expressions:
        return 0

    punct = ".,!?;:\"'()[]<>"
    c = 0

    for tok in text.split():           
        t = tok.casefold().strip(punct)   
        if t and t in expressions:     
            c += 1

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

def count_average_caracters(text: str) -> float:

    if not text:
        return 0.0

    punct = ".,!?;:\"'()[]<>…—-"
    tokens = [tok.strip(punct) for tok in text.split()]
    tokens = [t for t in tokens if t] 

    if not tokens:
        return 0.0

    total_chars = sum(len(t) for t in tokens)
    return total_chars / len(tokens)


def count_polar_expressions(text: str,
                            pos_words: set[str],
                            neg_words: set[str],
                            intensifiers: set[str],
                            negations: set[str],
                            links: set[str],
                            pos_emojis: set[str],
                            neg_emojis: set[str]) -> tuple[int, int]:
    if not text:
        return 0, 0
    toks = tokenize(text)
    num_tokens = len(toks)
    pos = 0
    neg = 0
    joined = text.casefold()
    pos_emoticone = sum(joined.count(emoticone) for emoticone in pos_emojis)
    neg_emoticone = sum(joined.count(emoticone) for emoticone in neg_emojis)
    i = 0
    while i < num_tokens:
        current_token = toks[i]
        is_pos = current_token in pos_words
        is_neg = current_token in neg_words
        if is_pos or is_neg:
            j = i - 1
            flip = False
            while j >= 0 and (toks[j] in intensifiers or toks[j] in links or toks[j] in negations)  and (i-j < 4) \
                    and not (toks[j] in pos_words or toks[j] in neg_words):
                if toks[j] in negations:
                    flip = not flip
                j -= 1
                
            if is_pos:
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
    return pos + pos_emoticone, neg + neg_emoticone


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
    pos_words = lexiques["positives_words"]
    neg_words = lexiques["negatives_words"]
    intensifiers = lexiques["intensifiers"]
    negation_words = lexiques["negation_words"]
    link_words = lexiques["link_words"]

    print("positives_emojis:", len(emoticon_pos))
    print("negatives_emojis:", len(emoticon_neg))
    print("positives_words:", len(pos_words))
    print("negatives_words:", len(neg_words))
    print("intensifiers:", len(intensifiers))
    print("negations:", len(negation_words))
    print("links:", len(link_words))

    current = df["SentimentText"].astype(str)
    df["nb_mots"] = current.apply( lambda x : len(str(x).split()))
    df["nb_caracteres"]  = current.apply(lambda x: len(str(x)))
    df["nb_moy_caracteres"]  = current.apply(lambda x: count_average_caracters(str(x)))
    df["nbr_emoticones_positifs"] = current.apply(lambda x: count_occurrences_emojis(x, emoticon_pos))
    df["nbr_emoticones_negatifs"] = current.apply(lambda x: count_occurrences_emojis(x, emoticon_neg))
    df["nb_mots_pos"] = current.apply(lambda x: count_occurrences_words(x, pos_words))
    df["nb_mots_neg"] = current.apply(lambda x: count_occurrences_words(x, neg_words))
    df["nb_intensifieurs"] = current.apply(lambda x: count_occurrences_words(x, intensifiers))
    df["nb_mots_negation"] = current.apply(lambda x: count_occurrences_words(x, negation_words))
    df["nb_majuscules"] = current.apply(lambda x: count_all_caps(x))
    df["nb_mentions"] = current.apply(lambda x: count_tokens_starting_with(x, '#'))
    df["nb_hashtags"] = current.apply(lambda x: count_tokens_starting_with(x, '@'))
    df["nb_exclamations"] = current.apply(lambda x: str(x).count("!"))
    df["nb_questions"]    = current.apply(lambda x: str(x).count("?"))
    df["mots_repetes"] = current.apply(lambda x: has_repeated_word(x))
    res = current.apply(lambda x: count_polar_expressions(x, pos_words, neg_words, intensifiers, negation_words,link_words, emoticon_pos, emoticon_neg))
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

