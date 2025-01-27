import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Heuristics:
    @staticmethod
    def random_word_deletion(text: str) -> str:
        """Deletes random words from text.

        Args:
            text (str): The text.

        Returns:
            str: The text.
        """

        words = text.split()
        if len(words) > 1:
            random_idx = random.randint(0, len(words) - 1)
            words.pop(random_idx)

        if len(words) > 1:
            random_idx = random.randint(0, len(words) - 1)
            words.pop(random_idx)

        return " ".join(words)

    @staticmethod
    def random_sentence_deletion(sentences: list[str]) -> list[str]:
        """Deletes one random sentence from given sentences.

        Args:
            sentences (list[str]): List of sentences.

        Returns:
            list[str]: List of sentences.
        """

        random_idx = random.randint(0, len(sentences) - 1)
        sentences.pop(random_idx)
        return sentences

    @staticmethod
    def random_sentence_redundancy(sentences: list[str]) -> list[str]:
        """Adds one random sentence into sentences.

        Args:
            sentences (list[str]): List of sentences.

        Returns:
            list[str]: List of sentences.
        """

        random_idx = random.randint(0, len(sentences) - 1)
        sentences.append(sentences[random_idx])
        return sentences

    @staticmethod
    def swap_words_within_sentence(text: str) -> str:
        """Swaps two randoms words in a sentence.

        Args:
            text (str): The sentence.

        Returns:
            str: The sentence.
        """

        words = text.split()
        size = len(words)

        if size > 1:
            for _ in range(2):
                idx1: int = 0
                idx2: int = 0
                while idx1 == idx2:
                    idx1 = random.randint(0, size - 1)
                    idx2 = random.randint(0, size - 1)

                words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)


def apply_heuristics(report: str) -> str:
    """Generates nonfactual inconsistent report by applying some heuristics to the original report.

    Args:
        report (str): Original report.

    Returns:
        str: Distorted report.
    """

    sentences = report[:-1].split(",")
    total_sentences = len(sentences)

    idx = random.randint(0, total_sentences - 1)
    sentences[idx] = Heuristics.random_word_deletion(sentences[idx])

    idx = random.randint(0, total_sentences - 1)
    sentences[idx] = Heuristics.swap_words_within_sentence(sentences[idx])

    if total_sentences > 1:
        sentences = Heuristics.random_sentence_deletion(sentences)
        sentences = Heuristics.random_sentence_redundancy(sentences)

    sentence = ", ".join(sentences)
    sentence += "."
    rejected_answer: str = "###RESPONSE:\n" + sentence

    return rejected_answer


def get_most_similar_fields(
    img_feats: np.array,
    database: np.array,
    field_name: str,
    train_df: pd.DataFrame,
    is_train_set_passed: bool = False,
) -> str:
    """Selects top2 fields from train set based on similar image features.

    Args:
        img_feats (np.array): Features. Shape of [1, 1024].
        database (np.array): Database of train set image features. Shape of [N, 1024].
        field_name (str): The field to be selected based on similarity.
        train_df (pd.DataFrame): Train dataframe.
        is_train_set_passed (bool, optional): Whether train set is used in this applied function. Defaults to False.

    Returns:
        str: Field.
    """

    sim = cosine_similarity(img_feats, database)  # [1, N]
    if field_name == "findings" and is_train_set_passed:
        top2_indices = sim.argsort()[
            0, -3:-1
        ]  # We take next two similar reports disregarding self
    else:
        top2_indices = sim.argsort()[0, -2:]
    fields: list[str] = []
    for idx in top2_indices:
        fields.append(train_df[field_name].iloc[idx])

    return f"1. {fields[0]}\n2. {fields[1]}"


def prompt_gen_sft(row: pd.DataFrame) -> str:
    """Generates the prompt with response.

    Args:
        row (pd.DataFrame): Single row containing all the fields.

    Returns:
        str: Prompt.
    """

    prompt: str = (
        "###INSTRUCTION:\n"
        + "You are an expert radiologist. Generate FINDINGS section not exceeding 100 words by understanding details based on the given corresponding sections below.\n"
        + "PAST-REPORTS:\n"
        + f"{row['similar_past_reports']}\n"
        + "CONTEXTUAL-TAGS:\n"
        + f"{row['tags']}\n"
        + "IMPRESSIONS:\n"
        + f"{row['impression']}\n"
        + "###RESPONSE:\n"
        + f"{row['findings']}"
    )
    return prompt


def prompt_gen_po(row: pd.DataFrame) -> str:
    """Generates the prompt.

    Args:
        row (pd.DataFrame): Single row containing all the fields.

    Returns:
        str: Prompt.
    """

    # Removing random tokens from findings section
    findings_tokens = row["findings"].split()
    for _ in range(4):
        random_idx = random.randint(0, len(findings_tokens) - 1)
        if len(findings_tokens) >= 2:
            findings_tokens.pop(random_idx)
    findings = " ".join(findings_tokens)

    prompt: str = (
        "###INSTRUCTION:\n"
        + "You are an expert radiologist. Enhance and improve PREDICTED-FINDINGS section not exceeding 100 words by understanding details based on the given corresponding sections below.\n"
        + "PAST-REPORTS:\n"
        + f"{row['similar_past_reports']}\n"
        + "CONTEXTUAL-TAGS:\n"
        + f"{row['tags']}\n"
        + "IMPRESSIONS:\n"
        + f"{row['impression']}\n"
        + "PREDICTED-FINDINGS:\n"
        + f"{findings}\n"
    )
    return prompt
