import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import Swin_V2_B_Weights, swin_v2_b
from tqdm import tqdm

from utils import (
    apply_heuristics,
    get_most_similar_fields,
    prompt_gen_po,
    prompt_gen_sft,
)

tqdm.pandas()


class PreprocessIuxrayDataset:
    def __init__(
        self,
        train_filepath: str,
        val_filepath: str,
        test_filepath: str,
        sft_train_filepath: str,
        sft_val_filepath: str,
        sft_test_filepath: str,
        po_train_filepath: str,
        po_val_filepath: str,
        po_test_filepath: str,
    ) -> None:
        """Initializes PreprocessIuxrayDataset class.

        Args:
            train_filepath (str): Path of train csv file.
            val_filepath (str): Path of val csv file.
            test_filepath (str): Path of test csv file (mainly the predicted test set from embryonic model).
            sft_train_filepath (str): Path to save sft train csv file.
            sft_val_filepath (str): Path to save sft val csv file.
            sft_test_filepath (str): Path to save sft test csv file.
            po_train_filepath (str): Path to save po train csv file.
            po_val_filepath (str): Path to save po val csv file.
            po_test_filepath (str): Path to save po test csv file.
        """

        self.train_filepath = train_filepath
        self.val_filepath = val_filepath
        self.test_filepath = test_filepath

        self.sft_train_filepath = sft_train_filepath
        self.sft_val_filepath = sft_val_filepath
        self.sft_test_filepath = sft_test_filepath

        self.po_train_filepath = po_train_filepath
        self.po_val_filepath = po_val_filepath
        self.po_test_filepath = po_test_filepath

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        layers = list(model.children())[:-1]
        self.model = nn.Sequential(*layers)
        for p in self.model.parameters():
            p.requires_grad = False
        self.transforms = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()

    def clean(self, text: str) -> str:
        """Cleans the text removing unnecessary spaces.

        Args:
            text (str): Text.

        Returns:
            str: Cleaned text.
        """

        return text.replace(" , ", ", ").replace(" . ", ". ").replace(" .", ".")

    def extract_img_features(self, image_paths: str) -> np.array:
        """Extracts image features.

        Args:
            image_paths (str): Paths of image.

        Returns:
            np.array: Features extracted. Shape of [1, 1024].
        """

        img = Image.open(f"datasets/iuxray/images/{image_paths.split(',')[0]}").convert(
            "RGB"
        )
        img = self.transforms(img)
        img = img.unsqueeze(0).to(self.device)  # [1, C, H, W]
        feats = self.model(img)  # [1, 1024]
        return feats.detach().cpu().numpy()

    def preprocess(self) -> None:
        """Preprocesses the dataset."""

        # Base preprocessing ---------------------------
        train = pd.read_csv(self.train_filepath)
        val = pd.read_csv(self.val_filepath)
        test = pd.read_csv(self.test_filepath)

        train.drop(
            ["problem", "anatomy", "observation", "uncertainty"], inplace=True, axis=1
        )
        val.drop(
            ["problem", "anatomy", "observation", "uncertainty"], inplace=True, axis=1
        )
        test.drop(
            ["problem", "anatomy", "observation", "uncertainty"], inplace=True, axis=1
        )

        train["findings"] = train["findings"].progress_apply(self.clean)
        train["impression"] = train["impression"].progress_apply(self.clean)

        val["findings"] = val["findings"].progress_apply(self.clean)
        val["impression"] = val["impression"].progress_apply(self.clean)

        test["findings"] = test["findings"].progress_apply(self.clean)
        test["impression"] = test["impression"].progress_apply(self.clean)
        test["prediction_fg"] = test["prediction_fg"].progress_apply(self.clean)

        # Populate test and val set with top-2 retreived findings, impression and tags from train set based on image similarity ---------------------------
        self.model.to(self.device)

        train["img_feats"] = train["image_paths"].progress_apply(
            self.extract_img_features
        )
        val["img_feats"] = val["image_paths"].progress_apply(self.extract_img_features)
        test["img_feats"] = test["image_paths"].progress_apply(
            self.extract_img_features
        )

        database = np.array(train["img_feats"].values.tolist()).squeeze(1)  # [N, 1024]

        val["similar_past_reports"] = val["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "findings", train)
        )
        val["tags"] = val["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "tags", train)
        )
        val["impression"] = val["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "impression", train)
        )

        test["similar_past_reports"] = test["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "findings", train)
        )
        test["tags"] = test["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "tags", train)
        )
        test["impression"] = test["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "impression", train)
        )

        train["similar_past_reports"] = train["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "findings", train, True)
        )
        train["tags"] = train["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "tags", train)
        )
        train["impression"] = train["img_feats"].progress_apply(
            lambda x: get_most_similar_fields(x, database, "impression", train)
        )

        train.drop(["img_feats"], inplace=True, axis=1)
        val.drop(["img_feats"], inplace=True, axis=1)
        test.drop(["img_feats"], inplace=True, axis=1)

        # Sft ---------------------------
        train_sft = train.copy(deep=True)
        val_sft = val.copy(deep=True)
        test_sft = test.copy(deep=True)

        train_sft["prompt"] = train_sft.progress_apply(prompt_gen_sft, axis=1)
        val_sft["prompt"] = val_sft.progress_apply(prompt_gen_sft, axis=1)

        train_sft.to_csv(self.sft_train_filepath, index=False)
        val_sft.to_csv(self.sft_val_filepath, index=False)
        test_sft.to_csv(self.sft_test_filepath, index=False)

        print(
            f"Splits info: train_size={len(train_sft)} | val_size={len(val_sft)} | test_size={len(test_sft)}"
        )
        print(
            f"Saved {self.sft_train_filepath}, {self.sft_val_filepath}, {self.sft_test_filepath} files"
        )

        # Po ---------------------------
        train_po = train.copy(deep=True)
        val_po = val.copy(deep=True)
        test_po = test.copy(deep=True)

        train_po["prompt"] = train_po.progress_apply(prompt_gen_po, axis=1)
        train_po["rejected"] = train_po["findings"].progress_apply(apply_heuristics)
        train_po["chosen"] = train_po["findings"].progress_apply(
            lambda x: "###RESPONSE:\n" + x 
        )

        val_po["prompt"] = val_po.progress_apply(prompt_gen_po, axis=1)
        val_po["rejected"] = val_po["findings"].progress_apply(apply_heuristics)
        val_po["chosen"] = val_po["findings"].progress_apply(
            lambda x: "###RESPONSE:\n" + x 
        )

        train_po.to_csv(self.po_train_filepath, index=False)
        val_po.to_csv(self.po_val_filepath, index=False)
        test_po.to_csv(self.po_test_filepath, index=False)

        print(
            f"Splits info: train_size={len(train_po)} | val_size={len(val_po)} | test_size={len(test_po)}"
        )
        print(
            f"Saved {self.po_train_filepath}, {self.po_val_filepath}, {self.po_test_filepath} files"
        )


def main() -> None:
    """Executes the main flow."""

    PreprocessIuxrayDataset(
        "datasets/iuxray/train.csv",
        "datasets/iuxray/val.csv",
        "datasets/iuxray/test_p.csv",
        "datasets/iuxray/sft_train.csv",
        "datasets/iuxray/sft_val.csv",
        "datasets/iuxray/sft_test.csv",
        "datasets/iuxray/po_train.csv",
        "datasets/iuxray/po_val.csv",
        "datasets/iuxray/po_test.csv",
    ).preprocess()


if __name__ == "__main__":
    main()
