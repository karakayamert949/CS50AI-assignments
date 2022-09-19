import nltk
import sys
import numpy as np
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary = {}

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as ofile:
            dictionary[file] = ofile.read()

    return dictionary

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    tokenizedDocument = nltk.tokenize.word_tokenize(document)
    document = [word for word in tokenizedDocument if word.isalpha()]
    return document

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    countDict = dict()
    uniqueWords = set(sum(documents.values(), []))
    for word in uniqueWords:
        for content in documents.values():
            if word in content:
                try:
                    countDict[word] += 1
                except KeyError:
                    countDict[word] = 1

    numDocs = len(documents)
    idfDict = dict()
    for word, count in countDict.items():
        idfDict[word] = np.log(numDocs/count)

    return idfDict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfDict = dict()
    for word in query:
        for nameFile, content in files.items():
            tf = content.count(word)
            if tf:
                try:
                    tfidfDict[nameFile] += tf*idfs[word]
                except KeyError:
                    tfidfDict[nameFile] = tf*idfs[word]

    sortedScores = [word for word, val in sorted(
        tfidfDict.items(), key=lambda x: x[1], reverse=True)]
    return sortedScores[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scoreDict = dict()
    for sentence, words in sentences.items():
        score = 0
        for word in query:
            if word in words:
                try:
                    score += idfs[word]
                except KeyError:
                    score = idfs[word]
        if score:
            density = words.count(word)/len(words)
            scoreDict[sentence] = (score, density)

    sortedSentences = [sentence for sentence, pair in sorted(
        scoreDict.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]
    return sortedSentences[:n]


if __name__ == "__main__":
    main()
