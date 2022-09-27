For this project, I was tasked with the following: Implement a deep learning model that expands single variable polynomials. The model should take a factorized sequence and predict the expanded sequence of the polynomial. For example,
- given "-n*(-n-18)" predict "n**2+18*n"
- given "4*y*(y-22)" predict "4*y**2-88*y"
- given "2*(31-7*cos(h))*cos(h)" predict "-14*cos(h)**2+62*cos(h)"

I was given the training data and a main.py file with a few functions I should not change. One restriction was that the model should NOT exceed 5M parameters.

For my submission, I created a relatively small transformer model. The encoder and decoder blocks each had one layer to keep the number of parameters under the limit (as in, I didn't stack encoders/decoders as is common for larger models). The self-attention blocks in the encoder and decoder each had 8 heads. I tokenized inputs and outputs at the character level, except for [cos, sin, tan] which were each their own token.

My model achieves 80% accuracy on the validation set. I do not have the final test set, so I don't know the final performance of the model. However, I know my performance on the test set was > 0.70 (the benchmark to pass).

Citations:

I borrowed heavily from two Pytorch tutorials:
1. Language Translation with NN.Transformer and TorchText: https://pytorch.org/tutorials/beginner/translation_transformer.html
2. NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
