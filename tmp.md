# 트랜스포머로 무엇을 할 수 있나요?

오늘코드 유튜브 영상 : https://youtu.be/xbQ0DIJA0Bc

Install the Transformers, Datasets, and Evaluate libraries to run this notebook.


```python
!pip install datasets evaluate transformers[sentencepiece]
```

    Collecting datasets
      Downloading datasets-2.14.6-py3-none-any.whl (493 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m493.7/493.7 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting evaluate
      Downloading evaluate-0.4.1-py3-none-any.whl (84 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.1/84.1 kB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting transformers[sentencepiece]
      Downloading transformers-4.35.0-py3-none-any.whl (7.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.9/7.9 MB[0m [31m18.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from datasets) (1.23.5)
    Requirement already satisfied: pyarrow>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (9.0.0)
    Collecting dill<0.3.8,>=0.3.0 (from datasets)
      Downloading dill-0.3.7-py3-none-any.whl (115 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.3/115.3 kB[0m [31m12.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets) (1.5.3)
    Requirement already satisfied: requests>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2.31.0)
    Requirement already satisfied: tqdm>=4.62.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (4.66.1)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets) (3.4.1)
    Collecting multiprocess (from datasets)
      Downloading multiprocess-0.70.15-py310-none-any.whl (134 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m16.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: fsspec[http]<=2023.10.0,>=2023.1.0 in /usr/local/lib/python3.10/dist-packages (from datasets) (2023.6.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets) (3.8.6)
    Collecting huggingface-hub<1.0.0,>=0.14.0 (from datasets)
      Downloading huggingface_hub-0.19.0-py3-none-any.whl (311 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m311.2/311.2 kB[0m [31m30.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from datasets) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from datasets) (6.0.1)
    Collecting responses<0.19 (from evaluate)
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers[sentencepiece]) (3.13.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers[sentencepiece]) (2023.6.3)
    Collecting tokenizers<0.15,>=0.14 (from transformers[sentencepiece])
      Downloading tokenizers-0.14.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.8/3.8 MB[0m [31m34.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting safetensors>=0.3.1 (from transformers[sentencepiece])
      Downloading safetensors-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m41.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting sentencepiece!=0.1.92,>=0.1.91 (from transformers[sentencepiece])
      Downloading sentencepiece-0.1.99-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m43.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from transformers[sentencepiece]) (3.20.3)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (23.1.0)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (3.3.2)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (4.0.3)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0.0,>=0.14.0->datasets) (4.5.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.19.0->datasets) (2023.7.22)
    Collecting huggingface-hub<1.0.0,>=0.14.0 (from datasets)
      Downloading huggingface_hub-0.17.3-py3-none-any.whl (295 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m295.0/295.0 kB[0m [31m31.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets) (2023.3.post1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.1->pandas->datasets) (1.16.0)
    Installing collected packages: sentencepiece, safetensors, dill, responses, multiprocess, huggingface-hub, tokenizers, transformers, datasets, evaluate
    Successfully installed datasets-2.14.6 dill-0.3.7 evaluate-0.4.1 huggingface-hub-0.17.3 multiprocess-0.70.15 responses-0.18.0 safetensors-0.4.0 sentencepiece-0.1.99 tokenizers-0.14.1 transformers-4.35.0
    


```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

    No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/629 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]





    [{'label': 'POSITIVE', 'score': 0.9598048329353333}]




```python
# pipeline?
```


```python
classifier(
    ["I've been waiting for a HuggingFace course my whole life.",
     "I hate this so much!",
     "행복하다",
     "즐겁다",
     "힘들다"]
)
```




    [{'label': 'POSITIVE', 'score': 0.9598048329353333},
     {'label': 'NEGATIVE', 'score': 0.9994558691978455},
     {'label': 'POSITIVE', 'score': 0.7440056800842285},
     {'label': 'POSITIVE', 'score': 0.6790698766708374},
     {'label': 'POSITIVE', 'score': 0.7637549638748169}]




```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

    No model was supplied, defaulted to facebook/bart-large-mnli and revision c626438 (https://huggingface.co/facebook/bart-large-mnli).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.15k [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/1.63G [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]





    {'sequence': 'This is a course about the Transformers library',
     'labels': ['education', 'business', 'politics'],
     'scores': [0.8445989489555359, 0.11197412759065628, 0.04342695698142052]}




```python
sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']
classifier(sequence_to_classify, candidate_labels)
#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}

```




    {'sequence': 'one day I will see the world',
     'labels': ['travel', 'dancing', 'cooking'],
     'scores': [0.9938651919364929, 0.003273811424151063, 0.0028610352892428637]}




```python
candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
classifier(sequence_to_classify, candidate_labels, multi_label=True)
#{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
# 'scores': [0.9945111274719238,
#  0.9383890628814697,
#  0.0057061901316046715,
#  0.0018193122232332826],
# 'sequence': 'one day I will see the world'}

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-ccf47dedf23a> in <cell line: 2>()
          1 candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
    ----> 2 classifier(sequence_to_classify, candidate_labels, multi_label=True)
          3 #{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
          4 # 'scores': [0.9945111274719238,
          5 #  0.9383890628814697,
    

    NameError: name 'sequence_to_classify' is not defined



```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

    No model was supplied, defaulted to gpt2 and revision 6c0e608 (https://huggingface.co/gpt2).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/665 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/548M [00:00<?, ?B/s]



    Downloading (…)neration_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    




    [{'generated_text': 'In this course, we will teach you how to take advantage of the information and techniques provided at the intersection of social, political and physical capital within India. The purpose of this course is to understand the ways India is changing for economic and political reasons.'}]




```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='skt/kogpt2-base-v2')
set_seed(42)
generator("점심 메뉴 추천,", max_length=30, num_return_sequences=3)
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.00k [00:00<?, ?B/s]



    Downloading pytorch_model.bin:   0%|          | 0.00/513M [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/2.83M [00:00<?, ?B/s]





    [{'generated_text': '점심 메뉴 추천, 저녁 메뉴 제안 등의 다양한 혜택을 제공한다고 밝혔다.\nCJ푸드빌이 운영하는 일식 레스토랑인 일식'},
     {'generated_text': '점심 메뉴 추천, 맥주 추천, 와인 추천, 식사 추천 등의 메뉴와 연계해 식사 후 간단한 와인 디너와 함께 디'},
     {'generated_text': '점심 메뉴 추천, 식사 후 디저트는 필수!\n서울 강남구 수서동에 위치한 ‘서울시푸드앤티’에서 운영하는 치킨전문'}]




```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=6,
)
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/762 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/353M [00:00<?, ?B/s]



    Downloading (…)neration_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/1.04M [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]


    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    




    [{'generated_text': 'In this course, we will teach you how to become better managers by reading over the course. This course can be bought in either eBook or pdf.'},
     {'generated_text': "In this course, we will teach you how to read. This isn't an isolated project. You will learn how to read with great skill as a"},
     {'generated_text': 'In this course, we will teach you how to get to work when you want it. We will give you an educational and educational overview and a walk'},
     {'generated_text': 'In this course, we will teach you how to do various types of work in different ways as well as how to organize work so you can learn where'},
     {'generated_text': 'In this course, we will teach you how to become an efficient, professional, and innovative entrepreneur, in this video series. This is an opportunity to'},
     {'generated_text': 'In this course, we will teach you how to use the tool on your own. If you take a walk to the test area in the centre you'}]




```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=5)
```

    No model was supplied, defaulted to distilroberta-base and revision ec58a5b (https://huggingface.co/distilroberta-base).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/480 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/331M [00:00<?, ?B/s]


    Some weights of the model checkpoint at distilroberta-base were not used when initializing RobertaForMaskedLM: ['roberta.pooler.dense.weight', 'roberta.pooler.dense.bias']
    - This IS expected if you are initializing RobertaForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]





    [{'score': 0.19619806110858917,
      'token': 30412,
      'token_str': ' mathematical',
      'sequence': 'This course will teach you all about mathematical models.'},
     {'score': 0.04052723944187164,
      'token': 38163,
      'token_str': ' computational',
      'sequence': 'This course will teach you all about computational models.'},
     {'score': 0.03301795944571495,
      'token': 27930,
      'token_str': ' predictive',
      'sequence': 'This course will teach you all about predictive models.'},
     {'score': 0.031941577792167664,
      'token': 745,
      'token_str': ' building',
      'sequence': 'This course will teach you all about building models.'},
     {'score': 0.024522852152585983,
      'token': 3034,
      'token_str': ' computer',
      'sequence': 'This course will teach you all about computer models.'}]




```python
unmasker_klue = pipeline("fill-mask", model="klue/bert-base")
unmasker_klue("한국인이 좋아하는 음식은 [MASK] 입니다.", top_k=3)
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/425 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/445M [00:00<?, ?B/s]


    Some weights of the model checkpoint at klue/bert-base were not used when initializing BertForMaskedLM: ['cls.seq_relationship.bias', 'bert.pooler.dense.weight', 'bert.pooler.dense.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


    Downloading (…)okenizer_config.json:   0%|          | 0.00/289 [00:00<?, ?B/s]



    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/248k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/495k [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/125 [00:00<?, ?B/s]





    [{'score': 0.1305919885635376,
      'token': 15764,
      'token_str': '비빔밥',
      'sequence': '한국인이 좋아하는 음식은 비빔밥 입니다.'},
     {'score': 0.07934065163135529,
      'token': 14564,
      'token_str': '불고기',
      'sequence': '한국인이 좋아하는 음식은 불고기 입니다.'},
     {'score': 0.066224105656147,
      'token': 6260,
      'token_str': '김치',
      'sequence': '한국인이 좋아하는 음식은 김치 입니다.'}]




```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision f2482bf (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/998 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/1.33G [00:00<?, ?B/s]


    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


    Downloading (…)okenizer_config.json:   0%|          | 0.00/60.0 [00:00<?, ?B/s]



    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]


    /usr/local/lib/python3.10/dist-packages/transformers/pipelines/token_classification.py:169: UserWarning: `grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to `aggregation_strategy="simple"` instead.
      warnings.warn(
    




    [{'entity_group': 'PER',
      'score': 0.9981694,
      'word': 'Sylvain',
      'start': 11,
      'end': 18},
     {'entity_group': 'ORG',
      'score': 0.9796019,
      'word': 'Hugging Face',
      'start': 33,
      'end': 45},
     {'entity_group': 'LOC',
      'score': 0.9932106,
      'word': 'Brooklyn',
      'start': 49,
      'end': 57}]




```python
ner("안녕하세요. 오늘코드 유튜브 채널입니다. 여기는 대한민국입니다.")
```




    []




```python
ner("Hello, I'm Korean")
```




    [{'entity_group': 'MISC',
      'score': 0.9982393,
      'word': 'Korean',
      'start': 11,
      'end': 17}]




```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at todaycode in seoul",
)
```

    No model was supplied, defaulted to distilbert-base-cased-distilled-squad and revision 626af31 (https://huggingface.co/distilbert-base-cased-distilled-squad).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/473 [00:00<?, ?B/s]



    Downloading model.safetensors:   0%|          | 0.00/261M [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/29.0 [00:00<?, ?B/s]



    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/436k [00:00<?, ?B/s]





    {'score': 0.739462673664093,
     'start': 33,
     'end': 51,
     'answer': 'todaycode in seoul'}




```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
"""
)
```

    No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e (https://huggingface.co/sshleifer/distilbart-cnn-12-6).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.80k [00:00<?, ?B/s]



    Downloading pytorch_model.bin:   0%|          | 0.00/1.22G [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]





    [{'summary_text': ' The number of engineering graduates in the United States has declined in recent years . China and India graduate six and eight times as many traditional engineers as the U.S. does . Rapidly developing economies such as China continue to encourage and advance the teaching of engineering . There are declining offerings in engineering subjects dealing with infrastructure, infrastructure, the environment, and related issues .'}]




```python
ko_summarizer = pipeline("summarization", model="gogamza/kobart-summarization")
ko_summarizer("상위 몇 개의 높은 확률을 띠는 토큰을 출력할지 top_k 인자를 통해 조절합니다. 여기서 모델이 특이한 <mask> 단어를 채우는 것을 주목하세요. 이를 마스크 토큰(mask token)이라고 부릅니다. 다른 마스크 채우기 모델들은 다른 형태의 마스크 토큰을 사용할 수 있기 때문에 다른 모델을 탐색할 때 항상 해당 모델의 마스크 단어가 무엇인지 확인해야 합니다. 위젯에서 사용되는 마스크 단어를 보고 이를 확인할 수 있습니다.")
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.18k [00:00<?, ?B/s]


    You passed along `num_labels=3` with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}. The number of labels wil be overwritten to 2.
    You passed along `num_labels=3` with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}. The number of labels wil be overwritten to 2.
    


    Downloading model.safetensors:   0%|          | 0.00/496M [00:00<?, ?B/s]


    You passed along `num_labels=3` with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}. The number of labels wil be overwritten to 2.
    


    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/446k [00:00<?, ?B/s]



    Downloading (…)olve/main/merges.txt:   0%|          | 0.00/177k [00:00<?, ?B/s]



    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/682k [00:00<?, ?B/s]



    Downloading (…)in/added_tokens.json:   0%|          | 0.00/4.00 [00:00<?, ?B/s]



    Downloading (…)cial_tokens_map.json:   0%|          | 0.00/111 [00:00<?, ?B/s]


    You passed along `num_labels=3` with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}. The number of labels wil be overwritten to 2.
    /usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1273: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(
    




    [{'summary_text': '위젯에서 사용된 위젯에서 사용되는 마스크 단어를 출력할지 top'}]




```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.42k [00:00<?, ?B/s]



    Downloading pytorch_model.bin:   0%|          | 0.00/301M [00:00<?, ?B/s]



    Downloading (…)neration_config.json:   0%|          | 0.00/293 [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/42.0 [00:00<?, ?B/s]



    Downloading (…)olve/main/source.spm:   0%|          | 0.00/802k [00:00<?, ?B/s]



    Downloading (…)olve/main/target.spm:   0%|          | 0.00/778k [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/1.34M [00:00<?, ?B/s]


    /usr/local/lib/python3.10/dist-packages/transformers/models/marian/tokenization_marian.py:197: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")
    




    [{'translation_text': 'This course is produced by Hugging Face.'}]




```python
ko_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
ko_translator("안녕하세요.")
```


    Downloading (…)lve/main/config.json:   0%|          | 0.00/1.39k [00:00<?, ?B/s]



    Downloading pytorch_model.bin:   0%|          | 0.00/312M [00:00<?, ?B/s]



    Downloading (…)neration_config.json:   0%|          | 0.00/293 [00:00<?, ?B/s]



    Downloading (…)okenizer_config.json:   0%|          | 0.00/44.0 [00:00<?, ?B/s]



    Downloading (…)olve/main/source.spm:   0%|          | 0.00/842k [00:00<?, ?B/s]



    Downloading (…)olve/main/target.spm:   0%|          | 0.00/813k [00:00<?, ?B/s]



    Downloading (…)olve/main/vocab.json:   0%|          | 0.00/1.72M [00:00<?, ?B/s]





    [{'translation_text': 'Hello.'}]




```python

```
