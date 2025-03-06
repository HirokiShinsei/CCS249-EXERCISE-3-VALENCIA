# CCS 249 - Natural Language Processing Unit 3 Exercise

## Exercise Title: Bi-Gram and Tri-Gram Model Training

### Overview
This exercise involves building and analyzing bi-gram and tri-gram models using text data from the Wikipedia page on **Quantum Mechanics**. The dataset is limited to the first 1000 words to emphasize working with constrained data and understanding n-gram language models.

### Objectives
- Learn the mechanics of bi-gram and tri-gram models.
- Practice preprocessing textual data for NLP tasks.
- Analyze n-gram frequency patterns and their implications.

### Dataset
- **Source**: Wikipedia page on [Quantum Mechanics](https://en.wikipedia.org/wiki/Quantum_mechanics).
- **Limitation**: First 1000 words of the page.

### Requirements
- **Programming Language**: Python.
- **Libraries**: 
  - [NLTK](https://www.nltk.org/)
  - [re](https://docs.python.org/3/library/re.html) (for regex-based text cleaning)
  - [Collections](https://docs.python.org/3/library/collections.html) (for frequency counting)
  - [Wikipedia](https://pypi.org/project/wikipedia-api/) (for fetching Wikipedia content)
- **Tools**: VS Code

### Instructions
1. **Text Extraction**:
   - Retrieve the content of the Quantum Mechanics Wikipedia page.
   - Limit the content to the first 1000 words.

2. **Preprocessing**:
   - Convert text to lowercase.
   - Remove special characters, numbers, and extra spaces.
   - Tokenize the text into words.

3. **N-Gram Generation**:
   - Generate bi-grams and tri-grams using sliding window techniques or NLTK.

4. **Frequency Analysis**:
   - Count the occurrences of each bi-gram and tri-gram.
   - Store and display the results as sorted frequency tables.

5. **Evaluation**:
   - Identify and document the most frequent n-grams.
   - Analyze patterns and their relevance to the Quantum Mechanics topic.

### Additional Resources
- [NLTK Documentation](https://www.nltk.org/)
- [Wikipedia API Guide](https://www.mediawiki.org/wiki/API:Main_page)
- [Python String Methods](https://docs.python.org/3/library/stdtypes.html#string-methods)



