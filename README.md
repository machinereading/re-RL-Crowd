# re-RL-Crowd

## Introduction
It is based on [RE-RL](https://github.com/unreliableXu/TensorFlow_RLRE) model. The following have been modified:
1. It dosen't use entity embeddings as the input features
2. Construct the bag of sentences randomly split sentences like batch data
3. Set the sampling size(number of pre-actions at training the policy network) to 1
4. Use twitter tokenizer to adjust Korean training data


## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://mrlab.kaist.ac.kr/contact)

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Acknowledgement
This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2013-0-00109, WiseKB: Big data based self-evolving knowledge base and reasoning platform)