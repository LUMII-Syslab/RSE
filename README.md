# Residual Shuffle-Exchange Networks: Official _TensorFlow_ Implementation

This repository contains the official _TensorFlow_ implementation of the following paper:

>**Residual Shuffle-Exchange Networks for Fast Processing of Long Sequences**
>
> by Andis Draguns, Emīls Ozoliņš, Agris Šostaks, Matīss Apinis, Kārlis Freivalds
>
> [[arXiv](https://arxiv.org/abs/2004.04662)]
>
>Abstract: _Attention is a commonly used mechanism in sequence processing, but it is of O(n²) complexity which prevents its application to long sequences. The recently introduced neural Shuffle-Exchange network offers a computation-efficient alternative, enabling the modelling of long-range dependencies in O(n log n) time. The model, however, is quite complex, involving a sophisticated gating mechanism derived from the Gated Recurrent Unit._
>
>_In this paper, we present a simple and lightweight variant of the Shuffle-Exchange network, which is based on a residual network employing GELU and Layer Normalization. The proposed architecture not only scales to longer sequences but also converges faster and provides better accuracy. It surpasses Shuffle-Exchange network on the LAMBADA language modelling task and achieves state-of-the-art performance on the MusicNet dataset for music transcription while using significantly fewer parameters._
>
>_We show how to combine Shuffle-Exchange network with convolutional layers establishing it as a useful building block in long sequence processing applications._

# Introduction

_Residual Shuffle-Exchange networks_ are a simpler and faster replacement for the recently proposed _Neural Shuffle-Exchange network_ architecture. It has O(*n* log *n*) complexity and enables processing of sequences up to a length of 2 million symbols where standard methods fail (e.g., attention mechanisms). The _Residual Shuffle-Exchange_ can serve as a useful building block for long sequence processing applications.

# Preview of results

Our paper describes _Residual Shuffle-Exchange networks_ in detail and provides full results on long binary addition, long binary multiplication, sorting tasks, the _LAMBADA_ question answering task and multi-instrument musical note recognition using the _MusicNet_ dataset.

Here are the accuracy results on the _[MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)_ transcription task of identifying the musical notes performed from audio waveforms (freely-licensed classical music recordings):

| **Model** | **Learnable parameters (M)** | **Average precision score (%)** |
| ------ | ------ | ------ |
| _cgRNN_ | 2.36 | 53.0 |
| _Deep Real Network_ | 10.0 | 69.8 |
| _Deep Complex Network_ | 17.14 | 72.9 |
| _Complex Transformer_ | 11.61 | 74.22 |
| **Residual Shuffle-Exchange network** | **3.06** | **78.02** |

Note: Our used model achieves state-of-the-art performance using significantly fewer parameters and the audio waveform directly compared to the previous two contenders that used specialised architectures with complex number representations of the Fourier-transformed waveform.

For a brief demonstration of our model's SOTA performance on MusicNet, see _[this video](https://youtu.be/RAu2p9xZiM4)_ on YouTube.

Here are the accuracy results on the _[LAMBADA](https://www.aclweb.org/anthology/P16-1144)_ question answering task of predicting a target word in its broader context (on average 4.6 sentences picked from novels):

| **Model** | **Learnable parameters (M)** | **Test accuracy (%)** |
| ------ | ------ | ------ |
| Random word from passage | - | 1.6 |
| _Gated-Attention Reader_ | unknown | 49.0 |
| _Neural Shuffle-Exchange network_ | 33 | 52.28 |
| **Residual Shuffle-Exchange network** | **11** | **54.34** |
| _Universal Transformer_ | 152 | 56.0 |
| _GPT-2_ | 1542 | 63.24 |
| Human performance | - | 86.0 |

Note: Our used model works faster and can be evaluated on 4 times longer sequences using the same amount of GPU memory compared to the _Shuffle-Exchange network_ model and on 128 times longer sequences than the _Universal Transformer_ model.

# What are _Residual Shuffle-Exchange networks_?

_Residual Shuffle-Exchange networks_ are a lightweight variant of the continuous, differentiable neural networks with a regular-layered structure consisting of alternating _Switch_ and _Shuffle_ layers that are _[Shuffle-Exchange networks](https://github.com/LUMII-Syslab/shuffle-exchange/blob/master/README.md)_.

The _Switch Layer_ divides the input into adjacent pairs of values and applies a _Residual Switch Unit_, a learnable 2-to-2 function, to each pair of inputs producing two outputs, employing _GELU_ and _Layer Normalization_.

Here is an illustration of a _Residual Switch Unit_, which replaces the _Switch Unit_ from _Shuffle-Exchange networks_:

**![](assets/readme-residual_shuffle_exchange_switch_operations.png)**

The _Shuffle Layer_ follows where inputs are permuted according to a perfect-shuffle permutation (i.e., how a deck of cards is shuffled by splitting it into halves and then interleaving them) – a cyclic bit shift rotating left in the first part of the network and (inversely) rotating right in the second part.

The _Residual Shuffle-Exchange network_ is organized in blocks by alternating these two kinds of layers in the pattern of the _Beneš network_. Such a network can represent a wide class of functions including any permutation of the input values. 
 
Here is an illustration of a whole _Residual Shuffle-Exchange network_ model consisting of two blocks with 8 inputs:

**![](assets/readme-residual_shuffle_exchange_model.png)**


## System requirements

- _Python_ 3.6 or higher.
- _TensorFlow_ 1.14.0.

## Running the experiments

To start training the _Residual Shuffle-Exchange network_ on binary addition, run the terminal command:
```
python3 RSE_trainer.py
```

To select the sequence processing task for which to train the _Residual Shuffle-Exchange network_ edit the `config.py` file that contains various hyperparameter and other suggested setting options.

For the _MusicNet_ transcription task see the following:
```
...
"""
    Task configuration.
"""
...
# task = "musicnet"
# input_type = tf.float32
...
```
To download and parse the _MusicNet_ dataset, run:
```
wget https://homes.cs.washington.edu/~thickstn/media/musicnet.npz 
python3 -u resample.py musicnet.npz musicnet_11khz.npz 44100 11000
rm musicnet.npz 
python3 -u parse_file.py
rm musicnet_11khz.npz
```
This might take a while. After parsing the file, make sure that config.py contains the correct directory for the _MusicNet_ data. To test the trained model for the _MusicNet_ task on the test set, run tester.py. 

For the _LAMBADA_ question answering task see the following:
```
...
"""
    Task configuration.
"""
...
# task = "lambada"
# n_input = lambada_vocab_size
# n_output = 3
# n_hidden = 48*8
# #input_dropout_keep_prob = 1.0
# input_word_dropout_keep_prob = 0.95
# use_front_padding = True
# use_pre_trained_embedding = True
# disperse_padding = False
# label_smoothing = 0.1
# batch_size = 64
# bins = [256]
...
```
To download the _LAMBADA_ dataset see the original publication by [Paperno et al](https://www.aclweb.org/anthology/P16-1144).

To download the pre-trained _fastText_ 1M English word embedding see the [downloads section](https://fasttext.cc/docs/en/english-vectors.html) of the _FastText_ library website and extract to directory listed in the `config.py` file variable `base_folder` under “Embedding configuration”:
```
...
"""
    Embedding configuration
"""
use_pre_trained_embedding = False
base_folder = "/host-dir/embeddings/"
embedding_file = base_folder + "fast_word_embedding.vec"
emb_vector_file = base_folder + "emb_vectors.bin"
emb_word_dictionary = base_folder + "word_dict.bin"
...
```

To enable the pre-trained embedding change the `config.py` file variable `use_pre_trained_embedding` to `True`:
```
...
use_pre_trained_embedding = True
...
```

To start training the _Residual Shuffle-Exchange network_ use the terminal command:
```
python3 DNGPU_trainer.py
```

If you're running _Windows_, before starting training the _Residual Shuffle-Exchange network_ edit the `config.py` file to change the directory-related variables to _Windows_ file path format:
```
...
"""
    Local storage (checkpoints, etc).
"""
...
out_dir = ".\host-dir\gpu" + gpu_instance
model_file = out_dir + "\\varWeights.ckpt"
image_path = out_dir + "\\images"
...
"""
    MusicNet configuration
"""
musicnet_data_dir = ".\host-dir\musicnet\musicnet"
...
"""
    Lambada configuration
"""
lambada_data_dir = ".\host-dir\lambada-dataset"
...
"""
    Embedding configuration
"""
...
base_folder = ".\host-dir\embeddings"
embedding_file = base_folder + "fast_word_embedding.vec"
emb_vector_file = base_folder + "emb_vectors.bin"
emb_word_dictionary = base_folder + "word_dict.bin"
...
```

## Contact information

For help or issues using _Residual Shuffle-Exchange networks_, please submit a _GitHub_ issue.

For personal communication related to _Residual Shuffle-Exchange networks_, please contact Kārlis Freivalds ([karlis.freivalds@lumii.lv](mailto:karlis.freivalds@lumii.lv)).
