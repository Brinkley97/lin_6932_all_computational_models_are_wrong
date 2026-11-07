import spacy

import numpy as np
import pandas as pd

from tqdm import tqdm
from spacy import displacy
from collections import defaultdict
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertModel

from data_processing import DataProcessing

class FeatureExtractionFactory(ABC):
    """An abstract base class to create feature extraction classes."""

    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str = None, type_of_df: str = "Standard"):
        self.df_to_vectorize = df_to_vectorize
        self.col_name_to_vectorize = col_name_to_vectorize
        self.type_of_df = type_of_df
        self.vectorizer = None

    def __name__(self):
        return self.__class__.__name__

    def extract_text_to_vectorize(self):
        text_to_vectorize = DataProcessing.df_to_list(self.df_to_vectorize, self.col_name_to_vectorize, self.type_of_df)
        return text_to_vectorize

    def word_feature_extraction(self):
        pass

    def sentence_embeddings_extraction(self):
        pass

    def feature_scores(self):
        pass

class SpacyFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "Spacy Feature Extraction"

    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str = None, type_of_df: str = "Standard", embedding_model_name: str = "spacy_large"):
        super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)
        self.embedding_model_name = embedding_model_name
        self.embedding_models = {
            'spacy_small': "en_core_web_sm",
            'spacy_medium': "en_core_web_md",
            'spacy_large': "en_core_web_lg",
            'spacy_transformer': "en_core_web_trf",
        }
        self.nlp = spacy.load(self.embedding_models[self.embedding_model_name])

    def sentence_embeddings_extraction(self, attach_to_df: bool = True):
        """Extract sentence (Doc) vector embeddings using SpaCy"""
        text_to_vectorize = self.extract_text_to_vectorize()
        sentence_embeddings = []
        
        # Check if using transformer model
        is_transformer = self.embedding_model_name == 'spacy_transformer'
        
        # Debug: Check what components are in the pipeline
        print(f"Pipeline components: {self.nlp.pipe_names}")
        
        for idx, sentence in enumerate(tqdm(text_to_vectorize)):
            if not isinstance(sentence, float) and pd.notna(sentence):
                doc = self.nlp(sentence)
                
                if is_transformer:
                    try:
                        # Method 1: Try using doc._.trf_data
                        if hasattr(doc._, 'trf_data') and doc._.trf_data is not None:
                            tensors = doc._.trf_data.tensors
                            if len(tensors) > 0 and tensors[0].numel() > 0:
                                embedding = tensors[0].mean(dim=0).cpu().numpy()
                                if idx < 3:  # Debug first few
                                    print(f"Method 1 success - embedding shape: {embedding.shape}")
                            else:
                                raise ValueError("Empty tensors")
                        else:
                            raise ValueError("No trf_data")
                            
                    except Exception as e1:
                        # Method 2: Manual token averaging
                        try:
                            if idx < 3:
                                print(f"Method 1 failed ({e1}), trying Method 2: manual token vectors")
                            
                            token_vectors = []
                            for token in doc:
                                if token.has_vector and token.vector.shape[0] > 0:
                                    token_vectors.append(token.vector)
                            
                            if token_vectors:
                                embedding = np.mean(token_vectors, axis=0)
                                if idx < 3:
                                    print(f"Method 2 success - embedding shape: {embedding.shape}")
                            else:
                                raise ValueError("No token vectors available")
                                
                        except Exception as e2:
                            if idx < 3:
                                print(f"Method 2 also failed ({e2}), using zeros")
                            embedding = np.zeros(768)
                else:
                    # For non-transformer models
                    embedding = doc.vector
                    if embedding.shape[0] == 0:
                        if idx < 3:
                            print(f"WARNING: Empty embedding for: {sentence[:50]}")
                
                sentence_embeddings.append(embedding)
            else:
                # For invalid sentences
                if is_transformer:
                    sentence_embeddings.append(np.zeros(768))
                else:
                    expected_size = self.nlp.meta.get('vectors', {}).get('width', 300)
                    sentence_embeddings.append(np.zeros(expected_size))

        if attach_to_df:
            self.df_to_vectorize[f"{self.col_name_to_vectorize} Embedding"] = sentence_embeddings
            return self.df_to_vectorize
        else:
            clean_embeddings = [emb for emb in sentence_embeddings if emb is not None]
            return np.array(clean_embeddings)
    
class SentenceTransformerFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "SentenceTransformer Feature Extraction"

    def __init__(
        self,
        df_to_vectorize: pd.DataFrame,
        col_name_to_vectorize: str = None,
        type_of_df: str = "Standard",
        embedding_model_name: str = "st_mini_lm"
    ):
        super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)

        self.embedding_models = {
            'st_mpnet_base': 'sentence-transformers/all-mpnet-base-v2',
            'st_distilroberta': 'sentence-transformers/all-distilroberta-v1',
            'st_minilm_l12': 'sentence-transformers/all-MiniLM-L12-v2',
            'st_minilm_l6': 'sentence-transformers/all-MiniLM-L6-v2',
        }

        self.embedding_model_name = embedding_model_name
        model_name = self.embedding_models[self.embedding_model_name]

        # Load model
        self.st = SentenceTransformer(model_name)

    def sentence_embeddings_extraction(self, attach_to_df: bool = True):
        """
        Extract sentence embeddings using SentenceTransformer

        Parameters
        ----------
        attach_to_df : bool, default True
            If True, attach embeddings to dataframe. Otherwise return numpy array

        Returns
        -------
        pd.DataFrame or np.ndarray
        """

        text_to_vectorize = self.extract_text_to_vectorize()

        # ✅ Core operation (from docs)
        embeddings = self.st.encode(text_to_vectorize, show_progress_bar=True)

        if attach_to_df:
            self.df_to_vectorize[f"{self.col_name_to_vectorize} Embedding"] = list(embeddings)
            return self.df_to_vectorize
        else:
            return embeddings