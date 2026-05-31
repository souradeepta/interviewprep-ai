# NLP (Pre-LLM Foundations)

8 concepts from text preprocessing to information retrieval — the foundations that underpin modern LLMs.

## Prerequisites
- Python, numpy, basic probability
- Familiarity with neural networks (see ai/ section)

## Concepts
| # | Concept | Key Content | Notebook |
|---|---------|-------------|---------|
| 01 | [Text Preprocessing](concepts/01-text-preprocessing.md) | Tokenization, TF-IDF, stemming | [notebook](notebooks/01-text-preprocessing.ipynb) |
| 02 | [Word Embeddings](concepts/02-word-embeddings.md) | Word2Vec skip-gram, GloVe, analogies | [notebook](notebooks/02-word-embeddings.ipynb) |
| 03 | [RNNs and LSTMs](concepts/03-recurrent-neural-networks.md) | LSTM, GRU, vanishing gradients | [notebook](notebooks/03-recurrent-neural-networks.ipynb) |
| 04 | [Seq2Seq and Attention](concepts/04-seq2seq-attention.md) | Encoder-decoder, Bahdanau attention | [notebook](notebooks/04-seq2seq-attention.ipynb) |
| 05 | [BERT and Pretraining](concepts/05-bert-and-pretraining.md) | MLM, NSP, fine-tuning | [notebook](notebooks/05-bert-and-pretraining.ipynb) |
| 06 | [Text Classification](concepts/06-text-classification.md) | BERT fine-tuning, few-shot | [notebook](notebooks/06-text-classification.ipynb) |
| 07 | [Named Entity Recognition](concepts/07-named-entity-recognition.md) | BIO tags, CRF, sequence labeling | [notebook](notebooks/07-named-entity-recognition.ipynb) |
| 08 | [Information Retrieval](concepts/08-information-retrieval.md) | BM25, dense retrieval, re-ranking | [notebook](notebooks/08-information-retrieval.ipynb) |

## How NLP Connects to LLMs
- Text preprocessing (01) → tokenization in GPT/BERT
- Word embeddings (02) → token embeddings in transformers
- Attention (04) → the "Attention Is All You Need" breakthrough
- BERT pretraining (05) → the template for all modern LLMs
