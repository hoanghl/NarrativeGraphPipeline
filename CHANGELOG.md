### This file contains changes and modifications during developing.

## Sep 5, 13:34

- Apply template for **MultiGPU training** and **Optuna hparams search**
- Fix error: q_ids, c_ids.. not correct in module **dataset**
- Train with 2 answers
- Refactor some code segments
- Discard **file shard** method

## Aug 26, 23:47

- Relace current backbone by **CHIME**
- Combine module **Graph** into backbone **CHIME**

## Aug 25, 23:51

- Temporarily comment module **GraphReasoning**
- Replace module _PGN_ by _TransDecoder_, use **Schedule Sampling for Transformer**
- Move function _loss calculation_ into module **Decoder**
- Refactor code in _metric calculation_
- Fix MLE loss calculation method

## Aug 23, 10:29

- Add **Scheduled Sampling** and **Optimal Transport IPOT** back

## Aug 22, 18:01

- Replace **BertBasedDecoder** by **LSTM with PointerGenerator, no Sched Sampling**
- Increase _d_hid_ from **64** to **256**

## Aug 21, 23:01

- Remove some redundant _GRU_ layers in module _FineGrain_
- Change shape of _context_ from **[b, n_c, d_bert]** to **[b, n_c, l_c, d_bert]**
- Add **Weak Attention hidden representation** into module _Reasoning_
- Add some other modifications

## Aug 20, 22:25

- Discard _memory component_ in _Graph reasoning module_
- Modify some configurations
- Use **Hard-Easy 50-50 strategy** for dataset switching

## Aug 16, 22:25

- Refactor datamodule and dataset to truly uses data **v4.3**
- Refactor variable names: ques -> q, context -> c...
- Set _Memory Parameter_ as `requires_grad=True`
- Increase _iterations_ in _ipot_ to 400
- Fix some errors

## Aug 16 2021, 15:35

- Use data v4.3
- Remove BeamSearch, use Greedy in validation
- Use Linear learning rate scheduler
- Changes some configurations in **configs.yaml**

## Aug 3 2021, 10:27

- Add 2 modes: _debug_train_ and _debug_predict_
- Fix important errors

## Aug 2 2021, 14:25

- Add FF layer in module _FineGrain_ in get OT to avoid NaN loss
- Add _BeamSearch_ to valid in module _ans_infer_
- Refactor code: remove configs and combine all into one single file, remove folder 'src', refactor _utils_
- Add pad_index into CrossEntropy ignore index

## July 26 2021, 15:00

- Use OT loss
- Move/Refactor code
- Reduce 'l_c' to 12

## July 20 2021, 00:03

- Apply new PATH deployment strategy: Narrative_utils stores anthing and its path is in .env
- Change some field names

## June 28 2021, 23:03

- Add new Scheduled Sampling strategy
- Reformat some field names

## June 20 2021, 14:43

- Apply data _version 4_ for training: not removing stopwords, new HTML removing, start-end extraction
- Reduce **n_paras** to _5_, **seq_len_ans** to _15_, increase **seq_len_para** to _170_
- Reduce **n_nodes** and **n_edges** in Graph module respectively
- Officially use BertVocab instead of own vocab
- Use **parquet** data format instead **csv**
- Change name of fields in _dataset.py_: **ques** to **ques_ids**, **paras** to **context_ids**,...

## June 10 2021, 11:09

- Add metric calculation for valid
- Add module _Preprocess_
- Remove _torch.nn.Embedding_ layer and replace by _BertEmbedding_
- Train with _BertVocab_ instead own vocab (BertTokenizer uses its own to tokenize)
- Increase `seq_len_para` to **182** instead of _122_
- Format files

## May 30 2021, 14:40

- Huge improvements: Use **Pytorch Lightning** and _template_
- Add new attentive strategy for _Finegrain_
- Modify module _Memory_
- Apply _BeamSearch_ modified from _huggingface_ implementation
- Discard _GloVe_
- Replace class _Vocab_ by _modified BertTokenizer_

## May 20 2021, 22:27

- Change code folder name from 'modules' to '
- Add new graph-based memory architecture _MemoryGraph_ for reasoning
- Reuse module _FineGrain_ to embed question and paras
- Change _Trainer_ and _CustomDataset_ to align with new updates

## May 17 2021, 21:24

- Modify `configs.py` and fit it with data _version 3.0_
- Modify module _NarrativePipeline_: fit with new field names
- Modify `utils.py` of NarrativePipeline to fit with new field names, _data version 3_,...

## May 9 2021, 23:39

- Reimplement _BeamSearch_: Replace _max_depth_ and _max_breadth_ by _beam_size_
- Add **Nucleus Sampling** technique to _BeamSearch_
- Disable metric calculating fucntions. Intend to use built-in.
- Modify some components in _Transformer_
- Add field `ans1_loss` to DataLoader during training
- Use _BertTokenizer_ and remove some methods in class _Vocab_

## May 6 2021, 22:44

- Reorganize _configs.py_
- Inferring in Transformer is now totally independent flow
- Temporarily remove calculating metrics, just generating predicted answers only
- Add 2 more fields `ans1_plain` and `ans1_plain` as creating iterator. Those are for inferring only.
- Replace user-generated vocab by BERT vocab for creating data for training/eval/test.

## May 2 2021, 21:18

- Reimplement _BeamSearch_ by simplifiyng many steps
- Switching context now occurs forcibly after training an epoch
- Process in module "DataReading" now runs on single process
- Function _build_vocab_ adds special tokens to vocab file

## Apr 25 2021, 16:18

- Reimplement function calculating _ROUGE-L_, _BLEU-1_...
- Implement _Beam Search_
- Implement new answer inferring module namely _TransDecoder_ ans use _Beam Search_
- Add feature logging metric calculation in testing phase

## Apr 21 2021, 15:07

- Fix errors in processing _movie_ context in _DataReading_ module
- Reconstruct _DataReading_ module
- Replace **LSTM-based** by **BERT-based** _Embedding layer_
- Change arguments of _PGD's forward_ to fit with new _Embedding layer_
- Modify training process's variables in model to fit with new _Embedding layer_
- Reorganize _GraphReasoning_ module into new folder
- Use trick longer answer to calculate loss and 2 answers to calculate metrics

## Apr 19 2021, 10:53

- Change and reconstruct repo to fit with _GCN_ setting

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
