The code implementation can be given as :  
[transformer.py]( https://github.com/AmritanshuV/Numpy-Transformer/blob/main/transformer.py) - contains the seq2seq class.

[encoder_layer.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/layers/combined/decoder_layer.py) - is the encoder layer, building block for [encoder.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/encoder.py) .

[self_attention.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/layers/combined/self_attention.py) shows the implementation of self attention mechanism.

[decoder_layer.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/decoder.py) base implementation for [decoder.py](https://github.com/AmritanshuV/Numpy-Transformer/blob/main/decoder.py)




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
Installation & Usage:
Details on how to install and use the Numpy-Transformer would be here.
