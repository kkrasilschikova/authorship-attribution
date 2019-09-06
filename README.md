# authorship-attribution
Given a text predict its author.

#### Data Set Information
To decrease the bias and create a reliable authorship attribution dataset the following criteria have been chosen to filter out authors in Gdelt database: English language writing authors, authors that have enough books available (at least 5), 19th century authors. With these criteria 50 authors have been selected and their books were queried through Big Query Gdelt database. The next task has been cleaning the dataset due to OCR reading problems in the original raw form. To achieve that, firstly all books have been scanned through to get the overall number of unique words and each words frequencies. While scanning the texts, the first 500 words and the last 500 words have been removed to take out specific features such as the name of the author, the name of the book and other word specific features that could make the classification task easier. After this step, we have chosen top 10,000 words that occurred in the whole 50 authors text data corpus. The words that are not in top 10,000 words were removed while keeping the rest of the sentence structure intact. The entire book is split into text fragments with 1000 words each. We separately maintained author and book identification number for each one of them in different arrays. Text segments with less than 1000 words were filled with zeros to keep them in the dataset as well. 1000 words make approximately 2 pages of writing, which is long enough to extract a variety of features from the document. Each instance in the training set consists of a text piece of 1000 words and an author id attached. In the testing set, there is only the text piece of 1000 words to do authorship attribution. Training data consists of 45 authors and testing data has 50 information. %34 of testing data is the percentile of unknown authors in the testing set.

#### Attribute Information
Each instance consists of 1000 word sequences that are divided from the works of every author's book. In the training, the author id is also provided.

#### Origin
http://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution
GUNGOR, ABDULMECIT, Benchmarking Authorship Attribution Techniques Using Over A Thousand Books by Fifty Victorian Era Novelists, Purdue Master of Thesis, 2018-04 
