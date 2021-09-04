# CHANGELOG

### This file contains changes and modifications during developing.

## Sep 4, 14:40

- Add functionalities for _tuning_

## Aug 31, 17:49

- Reimplement model **Yi Tay**
- Discard **file shard** method

## Aug 27, 19:03

- Move loss to model
- Add functionality: OT loss, calculate loss with 2 targets
- Fix some errors

## Aug 24, 16:26

- Reimplement model **CHIME**

## July 10 2021, 16:57

- Use data v4.2
- Reorganize data (raw, processed, bert-base-uncased); move some environment-dependent paths to _.env_ file
- Add config _num_shards_
- Stop switching En-Hn after swithching ratio reaches 1
- Add some configurations

## July 8 2021, 12:07

- Add OT loss, move get_score to `utils.py`
- Use _Hard Softmax_
- Discard old Reasoning module. Use totally _CHIME_ instead

## June 28 2021, 0:48

- Fix WEAM: loss is calculated by spare_p instead of output
- Finish implementation of _do_predict_

## June 26 2021, 8:23

- Apply **M-WEAM**
- Fix error backward is too slow

## June 24 2021, 10:33

- Fix error in module _Memory_
- Modify _Scheduled Sampling_ in module _ans_infer_layer_ and _narrative_model_
- Modify loss calculating formulation

## June 21 2021, 23:54

- Change name of **seq*len*** to **len\_**
- Change method of calculating loss: shift right 'ans_1' by 1
- Increase nth epoch to start scheduled sampling to 20
- Invent new reasoning module based on TransEncoder, Memory and CoAttention
- Remove _PG_ module in _Decoder_

## June 19 2021, 11:46

- Apply data _version 4_ for training: not removing stopwords, new HTML removing, start-end extraction
- Reduce **n_paras** to _5_, **len_ans** to _15_, increase **len_para** to _170_
- Officially use BertVocab instead of own vocab
- Use **parquet** data format instead **csv**
- Change name of fields in _dataset.py_: **ques** to **ques_ids**, **paras** to **context_ids**,...
- Add _PointerGenerator_ mechanism on top of current _BertDecoder_
- _Teacher forcing_ ratio is now backed up into checkpoints and logged into _Tensorboard_

## June 14 2021, 21:51

- Apply _Adaptive Teacher forcing_
- Replace _BertDecoder_ in module _ans_infer_, use _GRU_ and _attention_ and _maxout pointer generator_ instead
- Modify and simplify module _reasoning_ and _memory_ to match with new module _ans_infer_

## June 13 2021, 16:59

- Replace _BeamSearchHuggingface_ by my own BeamSearch (this is the old one)
- Freeze BertModel of embedding, don't load pretrained weights of BertModel in Decoder
- Modify some configurations
- Reorganize: move _get_score()_ to _narrative_model_, move _BeamSearch_ to separated `generator.py`, move Dataset to separated file
- Fix error in _config.resume_from_checkpoint_
- Add module _calc_diff_ave_ to calculate average difference between datapoint during training
- Add _Preprocess_ module

## June 2 2021, 21:30

- Move all project to **PyLightning + Hydra template**
- Use new decoder modified from _hugginface_ Bert implementation
- Apply new CoAttention masking for _Memory_
- Apply _CustomSampler_ for _DataModule_

## May 17 2021, 15:57

- Attention mask in module _MemoryBased_ is _Linear_ module now instead of a tensor.
- Fix error in module _DataReading_

## May 16 2021, 18:05

- Modify paths in 'configs.py' to make them consistent and add checking existence with paths
- Reimplement module 'DataReading'

## May 16 2021, 11:16

- Modify module _DataReading_ to fit with old para separation strategy of original paper
- Increase 'len_para' and decrease 'n_paras' in 'configs.py'
- Apply data _version 3_ for training

## May 15 2021, 10:26

- Take _GloVe embedding_ back and modify class _Vocab_
- Edit some fields during training
- Enhance some args in 'configs.py'
- Add Embedding-enhanced module _BertBasedEmbd_
- Modify output in module _MemoryBased_: ouput now is not just from memory but combination of it with outputs from each para.

## May 14 2021, 10:26

- Add new reasoning module: _MemoryBased_ that is based on Memory network
- Modify in module _FineGrain_: convert from NN to Linear
- Modify module _TransDecoder_ and _NarrativePipeline_ to fit with new reasoning module

## May 11 2021, 16:32

- Add new implementation of _BeamSearch_
- Simplify `launcher.sh`
- Remove _nltk_-related and metric calculating related. Inferring results now are dumped into JSON file for evaluating.
- Modify _TransDecoder_ module: result from inferring are not applied masks.
- Add field `ans1_loss` to calculate loss for predicted answer

## May 3 2021, 10:10

- Modify minor in _configs_
- Reimplement _BeamSearch_ with some simplifications
- Use _FineGrain_ with modifications instead of traditional _Embedding_
- Add changes in training pipeline
- Change fields in processing during training pipeline
- Modify class _Vocab_: remove GloVe vectors, upgrade method _stoi_

## Apr 29 2021, 11:20

- Apply new _Beam Search_ from _huggingface_
- Add _Vectors_ to class _Vocab_
- Add some configs related to new _Beam Search_
- Remove old file _utils_origin_

## Apr 27 2021, 15:25

- Implement _Beam Search_ and add to _TransDecoder_ module
- Reimplement _get_score_ function to calculate _BLEU-1_, _BLEU-4_...
- Add feature writing metric scores during testing

## Apr 19 2021, 15:45

- Fix serious errors in _PGD_ about dimension and teacher forcing
- Add second answer to calculate loss and metrics

## Apr 19 2021, 11:11

- Update changes in _PGN_ of branch _dev-proposal_ to current branch

## Apr 6 2021, 14:45

- Apply parallel processing and shared memory to _DataReading_ module
- Back up processed context file => speed up processing time

## Apr 6, 2021, 14:45

- Change module _data_reading_ significantly:
  - Construct entire module into class, class _DataReading_
  - Define may to process story and movie script separatedly
  - Process context before process each entry => speed up processsing
  - Apply top-down golden paras selection instead of tree-based

## Mar 30, 2021, 16:35

- Change minor thing in _IAL_ to speed up training
- Fix some problems in _trigger_inference_

## Mar 28, 2021, 22:17

- Reduce vocab size to 10 times smaller: Use top 1000 most occured words from context and words from answer
- Change in accumulating loss (detaching it before accumulating)
- Contiguous and change dimensions at some points in _PGD_
- Change the way of calculating _Equation 2_, hence reduce time and memory prohibitively spent during training

## Mar 26, 2021, 20:35

- Reimplement _Dataset_ module
- Finish test a training batch

## Mar 25, 2021, 11:18

- Finish remaining modules for training, testing and evaluating.

## Mar 19, 2021, 18:05

- Finish implementing _PointerGeneratorDecoder_
- Add question tokens to global vocab of _PointerGeneratorDecoder_

## Mar 16, 2021, 12:54

- Finish module _FineGrain_, _CustomDataset_, _IAL_

## Mar1, 2021, 16:50

- Finish _FineGrain_ module with single question and paragraph as input
- Modify _config_ file

## Mar 8, 2021, 12:00

- Abandon idea of using trained Bert-based Paras Selector and back to idea proposed in paper _Simple and Effective_ - using TfIdf

## Mar 2, 2021

- Use _csv_ format instead of _pickle_ for data backup
- Finish implementing training and inferring module for _ParasSelection_

## Feb 21, 2022

- Use 'DataFrame' in _pandas_ to pickle data and for easily deal with 'Dataset' of _huggingface_
- Reformat the data to be backed up

## Feb 17, 2021

- Reorder resources of paths of backup file. From now, paths are stored in `configs.py`

## Feb 16 2021, 15:34 PM

- Finish 'data_reading' module.

## Feb 13 2021, 10:18 AM

- Add 'data_reading' module to read, initially preprocess and decompose documents into paragraphs

## Feb 3 2021, 11:00 AM

- Very first commit
- Add default files: `.gitignore`, `CHANGELOG.md`, `configs.py`, `launcher.sh` and `README.md`
