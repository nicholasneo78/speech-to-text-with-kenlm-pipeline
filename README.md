# Speech-to-Text with Kenlm Pipeline
A self-supervised speech-to-text pipeline that is finetuned on HuggingFace's wav2vec2 and wavlm pretrained models. The finetuned models produced from this pipeline is then evaluated with the use of `Kenlm` language model and beam search capability from the `pyctcdecode` libraries. This repository is centered around the ClearML MLOps Framework, but running the code on the local machine is also possible.  
  
## Introduction 
#### Research Papers  
  
[wav2vec2 Architecture](https://arxiv.org/abs/2006.11477)   
[WavLM Architecture](https://arxiv.org/abs/2110.13900)   
   
#### Tasks in this pipeline   
The tasks in this pipeline are as follows:  

<ol>
  <li>Data Preprocessing</li>
  <li>Building the language model using Kenlm</li>
  <li>Finetuning the pretrained wav2vec2 and wavlm models from HuggingFace</li>
  <li>Evaluation of the finetuned model with the Kenlm language model built</li>
</ol>
  
## Getting Started
