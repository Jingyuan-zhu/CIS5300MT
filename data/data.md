# Data for MS1
## Train and Dev
We will use Helsinki-NLP/opus-100 on Hugging Face: https://huggingface.co/datasets/Helsinki-NLP/opus-100
### Format
Datasets are in parquet format, and their original format is in dict style, e.g. 
```
{
 "translation":{
 "en": "It was the asbestos in here, that's what did it!",
 "es": "Fueron los asbestos aquí. ¡Eso es lo que ocurrió!"
 }
}
```
Because we focus on en-es one-way translation, the en part will be input, and es will be labeled output.
### Size
The OPUS-100's en-es subset has 1 million rows for training by default, and we will combine the 2,000 dev and test sets into one 4,000-row dev set for our project.
### Load
We can directly load this dataset through HF's datasets package since it's very clean.
```
from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-es")
```

## Test
We will use openlanguagedata/flores_plus on Hugging Face: https://huggingface.co/datasets/openlanguagedata/flores_plus
### Format
It's in parquet format as well, and its original format is in a different dict style by single language only, since it's a fully aligned, many-to-many benchmark.

For example, the first data point is:

```
{'id': 0,
 'iso_639_3': 'eng',
 'iso_15924': 'Latn',
 'glottocode': 'stan1293',
 'text': 'On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.',
 'url': 'https://en.wikinews.org/wiki/Scientists_say_new_medical_diagnostic_chip_can_sort_cells_anywhere_with_an_inkjet',
 'domain': 'wikinews',
 'topic': 'health',
 'has_image': 'yes',
 'has_hyperlink': 'yes',
 'last_updated': '1.0'}
```
We only need to do some simple processing to get a format similar to OPUS-100.
### Size
Since this is an evaluation dataset, there are only a development (dev) and a test set, with the dev subset containing 997 rows and the test subset containing 1012 rows. By combining them, we will have a 2009 row test set.
### Load
Similar to OPUS-100, we can directly load Flores+ using HF's datasets package:

```
from datasets import load_dataset

ds_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
ds_es = load_dataset("openlanguagedata/flores_plus", "spa_Latn")
```
And do some processing:
```
combined_text = {
 "en": ds_en['dev'][0]['text'],
 "es": ds_es['dev'][0]['text']
}
```
The first row will be organized as:
```
{'en': 'On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.', 'es': 'El lunes, los científicos de la facultad de medicina de la Universidad de Stanford anunciaron el invento de una nueva herramienta de diagnóstico que puede catalogar las células según su tipo: un pequeñísimo chip que se puede imprimir y fabricar con impresoras de inyección de uso corriente, por un posible costo de, aproximadamente, un centavo de dólar por cada uno.'}
```

## Preperation
All of them are cleaned and combined into 3 parquet files.