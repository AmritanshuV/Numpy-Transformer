Installation and usage:
Setting up the envoirnment for Transformers:
Setup virtual env.

Install the required packages: The installed required packages could be found here [requirements.txt](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/transformer/requirements.txt).
Please be aware! These packages are set according to the CUDA Driver and python version in your system.
For running it on HPCs(for eg Alex@NHR FAU), see the modules available and install the drives in the virtual env accordingly

The code implementation can be given as :  
[transformer.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/transformer/transformer.py) - contains the seq2seq class.
For running the code:
Go to the directory [transfomer](https://github.com/AmritanshuV/Numpy-Transformer/tree/main/transformer) and use "python transformer.py"



**Numpy-Transformer**
Numpy-Transformer is a numpy-based implementation of the Transformer architecture, one of the most influential models in Machine Translation and Natural Language Processing.

Main Components:
Encoder: Understands the input sequence and compresses the information into a context.

Decoder: Takes the encoded information and translates it into the desired sequence.

Self-Attention Mechanism: Allows the model to consider other words in the input sequence when encoding a particular word.

Multi-Head Attention: Splits the input into multiple heads to capture different types of relationships in the data.

Features:
🚀 Pure Numpy: No deep learning libraries required.
🔍 Attention Visualization: Understand how the model focuses on various parts of the input.
💡 Interpretable: Written with clarity in mind for educational purposes.
