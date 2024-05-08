# BioNLI
*We have used a competition dataset from CodaLab whose link is https://github.com/ai-systems/Task-2-SemEval-2024/tree/main
# The NLI4CT Task-2-SemEval-2024 corpus consists of 2,400 statements divided into three datasets for training, development, and testing. The training dataset contains 1,750 statements, the development dataset contains 208 statements, and the test dataset contains 5,667 statements.
# Models Description
# DeBERTa-v3-base-mnli-fever-anli
# DeBERTa-v3-base model, trained using 763,913 hypotheses and premises, outperforms most large models in the ANLI benchmark, with the DeBERTa v3 variation significantly outperforming earlier iterations. Hugging face link: https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
# BERT-base uncased
# BERT is a transformer architecture model pre-trained in English using a large corpus of data without human labeling. It uses raw texts for inputs and labels, with a masked language modeling (MLM) objective, ensuring no distinction between English and other languages, Hugging face link: https://huggingface.co/google-bert/bert-base-uncased
# DeBERTa-v3-base-mnli-fever-docnli-ling-2c
# DeBERTa-v3 is an NLI model trained on 8 datasets, making predictions on statement entailment using 1,279,665 hypothesis-premise pairs. The inclusion of DocNLI with extensive texts improves long-range thinking. Hugging face: https://huggingface.co/DeBERTa-v3-base-mnli-fever-docnli-ling-2c
# Flan-T5-base 
# Flan-T5 base model refines pre-trained models using labeled datasets, tokenizing, preprocessing, and gradient descent techniques, classifying new text, providing predictions, and enabling high-precision sentiment analysis and topic categorization, Hugging face link: https://huggingface.co/google/flan-t5-base
# DeBERTa-large-mnli
# DeBERTa is a modified version of BERT and RoBERTa models with two distinct strategies: disentangled attention mechanism and improved mask decoder. It calculates attention weights between words and accurately forecasts masked tokens, Hugging face link: https://huggingface.co/microsoft/deberta-large-mnli
# BioBERT-base-cased-v1.1-mnli
# The BioBERT large-cased-v1.1-mnli model is an NLP (natural language processing) model that is implemented in the Transformer library, often with Python, Hugging Face link: https://huggingface.co/dmis-lab/biobert-base-cased-v1.1-mnli
# BioLinkBERT-base
# LinkBERT is a transformer encoder model similar to BERT. It has been pre-trained on a large corpus of documents. The model has been improved by adding links between documents, such as hyperlinks and citation links. This enables the model to incorporate knowledge that is dispersed across multiple sources, Hugging face link: https://huggingface.co/michiyasunaga/BioLinkBERT-base
