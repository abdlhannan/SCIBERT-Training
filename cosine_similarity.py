from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import plotly.express as px
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import pandas as pd

import nltk
nltk.download('punkt')
"""# Comparing patent sentences with the corresponding content of doi paper."""
SENTENCETRANSFORMER_MODEL_NAME = 'SciBert_finetuning_BIOSSES'
model_save_path = './output/'+SENTENCETRANSFORMER_MODEL_NAME

model = SentenceTransformer(model_save_path)

PATH = './'
pairs_df = pd.read_excel(PATH + "Paper_Corpus.xlsx", sheet_name=0, header=2)
corpus_df = pd.read_excel(PATH + 'Paper_Corpus.xlsx', sheet_name=1)

"""This code extracts all the sentences from "Sentences" folder and the corresponding mapping number from the file number"""

files = glob.glob(PATH + "/Sentences/*.txt")
def create_sentence_df(file_list):
    sentences_df = pd.DataFrame(columns=['filename', 'sentence', 'mapping'])
    sentences = []
    mapping = []
    for file in file_list:
        with open(file, 'r') as fd:
            sentences.append(fd.readline())
            mapping.append(int(file.split('#')[1].split('_')[0].replace(',', '')))
    sentences_df['filename'] = file_list
    sentences_df['sentence'] = sentences
    sentences_df['mapping'] = mapping
    
    return sentences_df

sentences_df = create_sentence_df(files)

"""# This cell reads each file in the Sentences folder. Gets the mapping number e.g USPTO-Dokument #8,501,349_1_P_1 will have 8501349. Search that number inside the Paper_Corpus file, get the corresponding doi numbers and extract the texts related to that doi. Compute Cosine Similarity, based on the embeddings by the trained model, and saves the maximum cos similarity sentence.
Doi texts are converted to each sentence. These sentences will be the corpus of that corresponding patent sentence. Each sentence is converted into embeddings, and compared against the patent sentence. Maximum similarity sentence is saved inside the csv in the end

There was a misspelling of doi 10.1017/s1431927614012744. In the Corpus sheet it was 10.1017/S1431927614012744 but in pairs sheet it was 10.1017/s1431927614012744. So i manually changed it.
"""

scores_list = []
top_sentence_list = []
doi_list = []
for INDEX in tqdm(range(sentences_df.shape[0])):
  test = sentences_df.iloc[INDEX, :]['mapping']
  doi_many = pairs_df[pairs_df.iloc[:, 2] == test]['citation external id'].values
  paper_text = []
  query_sentence = [sentences_df.iloc[INDEX, :]['sentence']]
  embeddings1 = model.encode(query_sentence, convert_to_tensor=True)
  max_cos_similariy_ref = -np.Inf
  most_relevant_sentence_ref = ''
  doi_ref = None
  for each_doi in doi_many:
    each_doi_text = corpus_df[corpus_df['DOI'] == each_doi]['Paper_Text'].values.tolist()
    each_doi_sentences = nltk.tokenize.sent_tokenize(each_doi_text[0])
    embeddings2 = model.encode(each_doi_sentences, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    descending_order_idx = reversed(np.argsort(cosine_scores[0, :].cpu()))
    max_cos_similariy = cosine_scores[0][descending_order_idx[0]]
    most_relevant_sentence = each_doi_sentences[descending_order_idx[0]]
    if max_cos_similariy > max_cos_similariy_ref:
      max_cos_similariy_ref = max_cos_similariy
      most_relevant_sentence_ref = most_relevant_sentence
      doi_ref = each_doi
  scores_list.append(max_cos_similariy_ref.cpu().numpy().tolist())
  top_sentence_list.append(most_relevant_sentence_ref)
  doi_list.append(doi_ref)



sentences_df['cosine_similarity'] = scores_list
sentences_df['doi'] = doi_list
sentences_df['top_sentence'] = top_sentence_list
sentences_df.to_csv(PATH+'scibert_pretrained_finetuned__pred_v1.csv', index=False)

"""Due to massive sizes of sentences, it cannot be shown on the graph. One way to intepret this graph can be, seeing the mapping patent number and checking the sentences manually"""
#df = px.data.tips()
fig = px.histogram(sentences_df, x="cosine_similarity",  color = 'mapping', hover_data=[sentences_df.doi], hover_name=sentences_df.doi)
fig.show()
