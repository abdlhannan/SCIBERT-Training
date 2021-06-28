
import nltk
nltk.download('punkt')
import pandas as pd
import nltk
from tqdm import tqdm

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
import gzip
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd
import glob
import pandas as pd

from tqdm import tqdm
import numpy as np

"""# PATH VARIABLES
These variables contain several paths which will be used by the script.
"""

OUTPUT_DIR_SCI_BERT = "./output/"
OUTPUT_MODEL_NAME = "SCIBERT_pretrained_on_patentdata"
TRAIN_PATH = "./datafinetuning_train.txt"
VAL_PATH = "./datafinetuning_val.txt"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
RANDOM_SEED = 0

"""# Script to create datafinetuning_train.txt and datafinetuning_val.txt files which will be used by the "Pre-training" of SCIBERT model.
Note: This cell is note required if these two files already exists.
"""


corpus = pd.read_excel("./Data_Fine-tuning.xlsx")
corpus = corpus.sample(frac=1, random_state=RANDOM_SEED)
corpus = corpus.dropna()
train_length = corpus.shape[0] * 0.80
val_length = 1 - train_length
train_list = []
dev_list = []
for i in tqdm(range(corpus.shape[0])):
    if i <= train_length:
      for column in ['TTL', 'ABST', 'ACLM']:
          sentences = nltk.tokenize.sent_tokenize(corpus.loc[i, column])
          for sentence in sentences:
              if sentence[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']: # Deleting the numbering in front of each sentence.
                  train_list.append(sentence)
    else:
      for column in ['TTL', 'ABST', 'ACLM']:
          sentences = nltk.tokenize.sent_tokenize(corpus.loc[i, column])
          for sentence in sentences:
              if sentence[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']: # Deleting the numbering in front of each sentence.
                  dev_list.append(sentence)

print("Total training sentences: ", len(train_list))
textfile = open(TRAIN_PATH, "w")
for element in train_list:
    textfile.write(element + "\n")
textfile.close()

print("Total validation sentences: ", len(dev_list))
textfile = open(VAL_PATH, "w")
for element in dev_list:
    textfile.write(element + "\n")
textfile.close()

"""# MASK LANGUAGE MODELLING task for Pre training of SCIBERT MODEL

NOTE: model will be saved in this directory: ./output/SCIBERT_pretrained_on_patentdata

Due to GPU Limitations, the max length of sentences will be 100. If sentence exceeds 100, the rest of sentence will be truncated.
"""


model_name = SCIBERT_MODEL
per_device_train_batch_size = 64

save_steps = 10               #Save model every 1k steps
num_train_epochs = 6            #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
do_whole_word_mask = True       #If set to true, whole words are masked
mlm_prob = 15                   #Probability that a word is replaced by a [MASK] token

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


output_dir = OUTPUT_DIR_SCI_BERT+"{}".format(OUTPUT_MODEL_NAME)
print("Save checkpoints to:", output_dir)


##### Load our training datasets

train_sentences = []
train_path = TRAIN_PATH
with gzip.open(train_path, 'rt', encoding='utf8') if train_path.endswith('.gz') else  open(train_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

print("Train sentences:", len(train_sentences))

dev_sentences = []
dev_path = VAL_PATH
with gzip.open(dev_path, 'rt', encoding='utf8') if dev_path.endswith('.gz') else open(dev_path, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            dev_sentences.append(line)

print("Dev sentences:", len(dev_sentences))

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None


##### Training arguments

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=100,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")

MODEL_PATH = "./output/SCIBERT_pretrained_on_patentdata"

"""# Script to convert Transformer model into Sentence Transformer model and train it for a STS(Sentence Similarity Task) using labelled dataset.
Name of sentence transformer model will be SciBert_finetuning_BIOSSES. 

Note: Model will be saved also in ./drive/MyDrive/Fiverr/sci_bert_training/output/
"""



SENTENCETRANSFORMER_MODEL_NAME = 'SciBert_finetuning_BIOSSES'
model_save_path = './output/'+SENTENCETRANSFORMER_MODEL_NAME

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = MODEL_PATH
# Read the dataset
train_batch_size = 8
num_epochs = 4


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')

train_samples = []
dataset = pd.read_csv('./BIOSSES-Dataset/BIOSSES_Pairs_Scores.csv')
dataset = dataset.sample(frac = 1)
dataset = dataset.dropna()
for i in range(dataset.shape[0]):
  text1 = dataset['Sentence 1'].iloc[i]
  text2 = dataset['Sentence 2'].iloc[i]
  #TAKING THE MAX 
  score_list = [int(dataset['Annotator A'].iloc[i]), int(dataset['Annotator B'].iloc[i]), \
                int(dataset['Annotator C'].iloc[i]), int(dataset['Annotator D'].iloc[i]), int(dataset['Annotator E'].iloc[i])]
  score = max(set(score_list), key = score_list.count)/4.0 # Normalize score to range 0 ... 1
  inp_example = InputExample(texts=[text1, text2], label=score)
  train_samples.append(inp_example)

logging.info("train samples: {}".format(len(train_samples)))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=None,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
