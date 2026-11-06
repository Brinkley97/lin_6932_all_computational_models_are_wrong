import os
import json

import pandas as pd


class DataProcessing:
    """Utilities for loading, saving, and preprocessing datasets."""

    # ==============================================================
    # PATH HELPERS
    # ==============================================================

    @staticmethod
    def load_base_data_path(script_dir: str) -> str:
        """
        Returns:
            path/to/data
        """
        return os.path.join(script_dir, "../data")

    # ==============================================================
    # FILE OPERATIONS
    # ==============================================================

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

            encodings_to_try = [
                encoding,
                "utf-8",
                "cp1252",
                "latin1",
                "iso-8859-1",
            ]

            last_exception = None

            for enc in encodings_to_try:

                try:
                    df = pd.read_csv(
                        path,
                        sep=sep,
                        encoding=enc,
                        **kwargs,
                    )

                    print(f"Loaded using encoding: {enc}")

                    return df

                except UnicodeDecodeError as e:
                    last_exception = e
                    continue

            raise last_exception


        elif file_type == "xlsx":
            return pd.read_excel(
                path,
                **kwargs,
            )

        elif file_type == "json":
            return pd.read_json(
                path
            )

        elif file_type == "txt":

            with open(
                path,
                "r",
                encoding=encoding
            ) as f:

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
        Standardize text and label columns across datasets.
        """

        df = df.copy()

        if label_map is not None:

            if label_col not in df.columns:
                raise ValueError(
                    f"Expected label column '{label_col}' not found."
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

        df = df.rename(columns=rename_map)

        if standardized_text_col not in df.columns:
            raise ValueError(
                f"Missing text column '{standardized_text_col}'."
            )

        if standardized_label_col not in df.columns:
            raise ValueError(
                f"Missing label column '{standardized_label_col}'."
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
    # FINANCIAL PHRASEBANK
    # ==============================================================
    def load_financial_phrasebank_dataset(
        script_dir: str,
        encoding: str,
    ) -> pd.DataFrame:
        """
        File:
            fpb.txt

        Format:
            sentence@label
        """

        fpb_path = os.path.join(
            script_dir,
            "project",
            "sentiment-financialPhraseBank",
            "sentiment-fpb.txt",
        )

        fpb_df = DataProcessing.load_from_file(
            fpb_path,
            file_type="txt",
            encoding=encoding,
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

        fpb_df = DataProcessing.standardize_columns(
            df=fpb_df,
            text_col="sentence",
            label_col="sentiment",
            standardized_text_col="Base Text Document",
            standardized_label_col="Sentiment Label",
        )

        fpb_df["Dataset Name"] = "financial_phrasebank"
        fpb_df["Data Split"] = "all"
        fpb_df["Original Source"] =  "https://huggingface.co/datasets/takala/financial_phrasebank"

        return fpb_df

    # ==============================================================
    # IMDB
    # ==============================================================
    def load_imdb_dataset(
        script_dir: str,
        data_split: str,
    ) -> pd.DataFrame:
        """
        data_split:
            train
            test
        """

        valid_splits = [
            "train",
            "test",
        ]

        if data_split not in valid_splits:
            raise ValueError(
                f"data_split must be one of {valid_splits}"
            )

        imdb_path = os.path.join(
            script_dir,
            "project",
            "sentiment-imdb",
            f"sentiment-imdb-{data_split}.csv",
        )

        imdb_df = DataProcessing.load_from_file(
            imdb_path,
            file_type="csv",
        )

        imdb_df = DataProcessing.standardize_columns(
            df=imdb_df,
            text_col="text",
            label_col="label",
            standardized_text_col="Base Text Document",
            standardized_label_col="Sentiment Label",
        )

        imdb_df["Dataset Name"] = "imdb"
        imdb_df["Data Split"] = data_split
        imdb_df["Original Source"] =  "https://huggingface.co/docs/transformers/v4.16.2/en/training"

        return imdb_df

    # ==============================================================
    # SEMEVAL 2019 SUGGESTION MINING
    # ==============================================================
    def load_semeval_2019_dataset(
        script_dir: str,
        data_split: str,
    ) -> pd.DataFrame:
        """
        data_split:
            train
            val
            test
        """

        valid_splits = [
            "train",
            "val",
            "test",
        ]

        if data_split not in valid_splits:
            raise ValueError(
                f"data_split must be one of {valid_splits}"
            )

        semeval_path = os.path.join(
            script_dir,
            "project",
            "suggestions-semeval_2019_task",
            f"suggestions-semeval-task_a-{data_split}.csv",
        )

        semeval_df = DataProcessing.load_from_file(
            semeval_path,
            file_type="csv",
        )

        text_col = semeval_df.columns[1]
        label_col = semeval_df.columns[-1]

        semeval_df = DataProcessing.standardize_columns(
            df=semeval_df,
            text_col=text_col,
            label_col=label_col,
            standardized_text_col="Base Text Document",
            standardized_label_col="Subjunctive Mood - Suggestions Label",
        )

        semeval_df["Dataset Name"] = "semeval_2019_suggestion_mining"
        semeval_df["Data Split"] = data_split
        semeval_df["Original Source"] =  "https://github.com/Semeval2019Task9/Subtask-A"

        return semeval_df

    # ==============================================================
    # YOUTUBE SUGGESTIONS
    # ==============================================================
    def load_youtube_dataset(
        script_dir: str,
    ) -> pd.DataFrame:

        yt_path = os.path.join(
            script_dir,
            "project",
            "suggestions-youTube",
            "suggestions-yt.csv",
        )

        yt_df = DataProcessing.load_from_file(
            yt_path,
            file_type="csv",
        )

        text_col = yt_df.columns[0]
        label_col = yt_df.columns[1]

        yt_df = DataProcessing.standardize_columns(
            df=yt_df,
            text_col=text_col,
            label_col=label_col,
            standardized_text_col="Base Text Document",
            standardized_label_col="Subjunctive Mood - Suggestions Label",
        )

        yt_df["Dataset Name"] = "youtube_suggestions"
        yt_df["Data Split"] = "all"
        yt_df["Original Source"] = "https://github.com/BurakaKrishna/Suggestion-Mining-Capstone/tree/master"

        return yt_df

    # ==============================================================
    # AMAZON SENTIMENT DATASET
    # ==============================================================
    def load_amazon_dataset(
        script_dir: str,
    ) -> pd.DataFrame:
        """
        File:
            sentiment-amazon.tsv

        Expected columns:
            review text column
            sentiment label column

        Automatically uses the first column as text and
        second column as the sentiment label.
        """

        amazon_path = os.path.join(
            script_dir,
            "project",
            "sentiment-amazon",
            "sentiment-amazon.tsv",
        )

        amazon_df = DataProcessing.load_from_file(
            amazon_path,
            file_type="csv",
            sep="\t",
            on_bad_lines='skip', 
            low_memory=False
        )

        text_col = amazon_df.columns[0]
        label_col = amazon_df.columns[1]

        amazon_df = DataProcessing.standardize_columns(
            df=amazon_df,
            text_col=text_col,
            label_col=label_col,
            standardized_text_col="Base Text Document",
            standardized_label_col="Sentiment Label",
        )

        amazon_df["Dataset Name"] = "amazon_sentiment"
        amazon_df["Data Split"] = "all"
        amazon_df["Original Source"] = (
            "https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz"
        )

        return amazon_df