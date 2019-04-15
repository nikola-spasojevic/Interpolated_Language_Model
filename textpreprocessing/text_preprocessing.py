#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
import pickle
import re

# Process the iput data to get a valid body of text
class TextPreprocessing:
    @staticmethod
    def corpus_gen(JOB_TITLE_DATA_DIR, TEST_CORPUS_DIR, TRAIN_CORPUS_DIR, EVAL_CORPUS_DIR):
        dataset = pd.read_csv(JOB_TITLE_DATA_DIR, header=None, names=['Job Titles'], encoding='utf8')

        stop_words = set(stopwords.words('english'))
        stop_words_french = set(stopwords.words('french'))
        stop_words = stop_words.union(stop_words_french)
        stop_words_german = set(stopwords.words('german'))
        stop_words = stop_words.union(stop_words_german)
        
        corpus, tokenized_corpus = [], []
        for i in range(len(dataset)):
            #Convert to lowercase
            text = dataset['Job Titles'][i].lower()

            # remove special characters and digits
            text=re.sub("(\\d|\\W)+"," ",text)

            #Convert to list from string
            text = text.split()

            #remove all stopwords from text body
            text = [word for word in text if word not in stop_words]

            # The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally 
            # related forms of a word to a common base form. (e.g beginning -> begin, cars -> car) 
            
            #Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text]

            #Stemming - could potentially improve the quality of proposed results
            # porter = PorterStemmer()
            # text = [porter.stem(word) for word in text if not word in stop_words]

            text = " ".join(text)
            corpus.append(text)

        # Train on 80% of the corpus 
        # Test the model on 10% - and use the results for tweaking
        # Evaluate the final model on the held out evaluation set (10%)
        spl_1 = int(90*len(corpus)/100)
        spl_2 = int(95*len(corpus)/100)
            
        train_corpus = corpus[:spl_1]
        test_corpus = corpus[spl_1:spl_2]
        eval_corpus = corpus[spl_2:]
        
        with open(TEST_CORPUS_DIR, 'wb') as output:
            pickle.dump(test_corpus, output)
            output.close()

        with open(TRAIN_CORPUS_DIR, 'wb') as output:
            pickle.dump(train_corpus, output)
            output.close()

        with open(EVAL_CORPUS_DIR, 'wb') as output:
            pickle.dump(eval_corpus, output)
            output.close()

def main():
    TextPreprocessing.corpus_gen(JOB_TITLE_DATA_DIR='../input/jobcloud_published_job_titles.csv',\
                                TEST_CORPUS_DIR='../bin/test_corpus.pkl', \
                                TRAIN_CORPUS_DIR='../bin/train_corpus.pkl', \
                                EVAL_CORPUS_DIR='../bin/eval_corpus.pkl') 

if __name__ == "__main__":
   main()
