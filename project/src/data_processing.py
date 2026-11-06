import os
import json

import pandas as pd


class DataProcessing:
    """Utilities for loading, saving, and preprocessing datasets."""

    # ==============================================================
    # FILE OPERATIONS
    # ==============================================================

    def load_base_data_path(notebook_dir: str) -> str:
        """Path to data/"""
        return os.path.join(notebook_dir, "../data")

    @staticmethod
    def load_from_file(
        path: str,
        file_type: str = "csv",
        sep: str = ",",
        encoding: str = "utf-8",
        **kwargs,
    ):
        """
        Load data from file.
        """

        if file_type == "csv":
            return pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                **kwargs
            )

        elif file_type == "xlsx":
            return pd.read_excel(
                path,
                **kwargs
            )

        elif file_type == "json":
            return pd.read_json(
                path
            )

        elif file_type == "txt":

            with open(path, "r", encoding=encoding) as f:
                lines = [
                    line.rstrip("\n")
                    for line in f
                ]

            return pd.DataFrame(
                {
                    "Text": lines
                }
            )

        raise ValueError(
            f"Unsupported file type: {file_type}"
        )

    @staticmethod
    def save_to_file(
        data,
        path: str,
        prefix: str,
        save_file_type: str,
        include_version: bool = True,
        append: bool = False,
        **kwargs,
    ) -> None:
        """
        Save data to file.
        """

        os.makedirs(path, exist_ok=True)

        next_number = None

        if include_version and not append:
            next_number = DataProcessing.get_next_file_number(
                path=path,
                prefix=prefix,
            )

        if save_file_type == "json":

            file_name = (
                f"{prefix}-v{next_number}.json"
                if include_version
                else f"{prefix}.json"
            )

            file_path = os.path.join(
                path,
                file_name,
            )

            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

        elif save_file_type in ["csv", ".csv"]:

            if append:

                file_name = f"{prefix}.csv"
                file_path = os.path.join(
                    path,
                    file_name,
                )

                file_exists = os.path.exists(
                    file_path
                )

                data.to_csv(
                    file_path,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )

            else:

                file_name = (
                    f"{prefix}-v{next_number}.csv"
                    if include_version
                    else f"{prefix}.csv"
                )

                file_path = os.path.join(
                    path,
                    file_name,
                )

                data.to_csv(
                    file_path,
                    index=False,
                )

        else:
            raise ValueError(
                f"Unsupported file type: {save_file_type}"
            )

    # ==============================================================
    # DATAFRAME HELPERS
    # ==============================================================

    @staticmethod
    def concat_dfs(
        dfs: list[pd.DataFrame],
        axis: int = 0,
        ignore_index: bool = True,
    ) -> pd.DataFrame:
        """
        Concatenate DataFrames.
        """

        return pd.concat(
            dfs,
            axis=axis,
            ignore_index=ignore_index,
        )

    @staticmethod
    def standardize_columns(
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        standardized_text_col: str = "Base Text Document",
        standardized_label_col: str = "Label",
        label_map: dict | None = None,
    ) -> pd.DataFrame:
        """
        Standardize dataset columns.

        Examples
        --------
        Sentiment:
            standardized_label_col="Sentiment Label"

        Suggestion Mining:
            standardized_label_col="Subjunctive Mood Label"
        """

        df = df.copy()

        if label_map is not None:

            if label_col not in df.columns:
                raise ValueError(
                    f"Expected label column "
                    f"'{label_col}' not found."
                )

            df[label_col] = (
                df[label_col]
                .astype(str)
                .str.lower()
                .map(label_map)
            )

        rename_map = {
            text_col: standardized_text_col,
            label_col: standardized_label_col,
        }

        df = df.rename(
            columns=rename_map
        )

        if standardized_text_col not in df.columns:
            raise ValueError(
                f"Missing text column: "
                f"{standardized_text_col}"
            )

        if standardized_label_col not in df.columns:
            raise ValueError(
                f"Missing label column: "
                f"{standardized_label_col}"
            )

        priority_cols = [
            standardized_text_col,
            standardized_label_col,
        ]

        remaining_cols = [
            col
            for col in df.columns
            if col not in priority_cols
        ]

        return df[
            priority_cols +
            remaining_cols
        ]

    # ==============================================================
    # DATASET LOADERS
    # ==============================================================

    @staticmethod
    def load_financial_phrasebank_dataset(
        script_dir: str,
        encoding: str
    ) -> pd.DataFrame:
        """
        Load Financial PhraseBank dataset.

        Expected format:
            Sentence@label

        Example:
            Revenue increased by 5%.@positive
        """

        # data/project/sentiment-financialPhraseBank/fpb.txt
        fpb_path = os.path.join(
            script_dir,
            "project",
            "sentiment-financialPhraseBank",
            "fpb.txt",
        )

        fpb_df = DataProcessing.load_from_file(
            fpb_path,
            file_type="txt",
            encoding=encoding
        )

        records = []

        for line in fpb_df["Text"]:

            if "@" not in line:
                continue

            sentence, sentiment = (
                line.rsplit("@", 1)
            )

            records.append(
                {
                    "sentence": sentence.strip(),
                    "sentiment": sentiment.strip(),
                }
            )

        fpb_df = pd.DataFrame(
            records
        )

        fpb_df = (
            DataProcessing.standardize_columns(
                df=fpb_df,
                text_col="sentence",
                label_col="sentiment",
                standardized_text_col="Base Text Document",
                standardized_label_col="Sentiment Label",
            )
        )

        fpb_df["Dataset Name"] = (
            "financial_phrasebank"
        )

        return fpb_df