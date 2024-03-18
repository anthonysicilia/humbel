# HumBEL: A Human-in-the-Loop Approach for Evaluating Demographic Factors of Language Models in Human-Machine Conversations
This is the resource repository for the paper "HumBEL: A Human-in-the-Loop Approach for Evaluating Demographic
Factors of Language Models in Human-Machine Conversations" to be published at EACL 2024.

## Announcements
Additional data annotations and code are forthcoming! Things to expect after I am done traveling and can get back to my local server:
1. updated code for ``scripts/gpt/get_results.py`` which contains interactions with huggingface models
2. A subset of (~150) annotated errors for Chat GPT (0613)

## Data
Automated test data (WC large) is available on huggingface: [anthonysicilia/wclarge](https://huggingface.co/datasets/anthonysicilia/wclarge)

The original sources of this data include:
- Word association data available from the [WAX dataset](https://aclanthology.org/2022.aacl-main.9.pdf)
- Test-based AoA data available from [Brysbaert et al.](https://link.springer.com/article/10.3758/s13428-016-0811-4), which is under a non-commercial creative commons license (see paper for details)

## Code
Code from the ```scripts''' directory can be used to re-create our results. As a standard order of operations, you will need to:
1. make the tests with ``gpt/make_tests.py`` or download the ready made test data at [anthonysicilia/wclarge](https://huggingface.co/datasets/anthonysicilia/wclarge).
2. get results with ``gpt/get_results.py``
3. interpret results with ``gpt/interpreter.py`` which parses the model answers
4. extract any features with ``gpt/features.py``
5. at this point, all dependencies should be create so you can use any of the other provided analysis scripts 


## Pearson Information
Clinical exam data from Pearson needs to be purchased directly from them, per our licensing agreement. [Here](https://www.pearsonassessments.com/store/usassessments/en/Store/Professional-Assessments/Speech-%26-Language/Clinical-Evaluation-of-Language-Fundamentals-%7C-Fifth-Edition/p/100000705.html) is a link to CELF5.

Note, any examples of test materials provided during discussion are adaptions of the original materials per publishing agreement with Pearson, Inc. While different, the examples are designed to convey similar qualitative insight to the reader; e.g., the LM prompt or types of errors made by the LM.

Clinical Evaluation of Language Fundamentals, Fifth Edition, CELF-5 Copyright © 2013 NCS Pearson, Inc. Repro- duced with permission. All rights reserved.

Clinical Evaluation of Language Fundamentals, Fifth Edi- tion, CELF-5 is a trademark, in the US and/or other countries, of Pearson Education, Inc. or its affiliates(s).
