# TextRank

A python implementation of TextRank. Based on [this post](https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/) by Prateek Joshi. I've modified a few things here and there.

## Getting started

1. Install `python 3.6` (haven't tested with other versions).

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. We use spacy to vectorize sentences and for that we need a particular [spacy model](https://spacy.io/usage/models), which we download by issuing:

    ```bash
    python -m spacy download en_core_web_lg
    ```

## Sanity check

```bash
python textrank.py
```

My output looks like this:

```
Rankings are as follows:
    Rank 1: Sentence #1 (Sentence two...)
    Rank 2: Sentence #2 (Sentence three...)
    Rank 3: Sentence #0 (Sentence one...)
```

## TODO
- [ ] add tests
- [ ] parametrize
- [ ] add functions to textrank files and directories
