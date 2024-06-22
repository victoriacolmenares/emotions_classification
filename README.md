# Proyecto de Fine-Tuning para Clasificación de Emociones

Este proyecto se centra en el fine-tuning de un modelo de clasificación de emociones utilizando el modelo base [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) y el dataset [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).

## Tabla de Contenidos

- [Introducción](#introducción)
- [Requisitos](#requisitos)
- [Dataset](#dataset)
- [Modelo](#modelo)

## Introducción

El propósito de este proyecto es mejorar un modelo de clasificación de emociones en inglés mediante el proceso de fine-tuning. Se busca crear un sistema capaz de clasificar textos en las categorias: `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`.

## Requisitos

- Python 3.7+
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [Evaluate](https://github.com/huggingface/evaluate)
- Otros paquetes necesarios están listados en `Pipefile`


## Dataset

El dataset utilizado es [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), que consiste en textos etiquetados con diferentes etiquetas de emociones.

### Preprocesamiento

- **Carga y División del Dataset:** El dataset se divide en conjuntos de entrenamiento y validación.
- **Tokenización:** Utilizamos el tokenizer del modelo `distilbert-base-uncased`.


## Modelo

- **Modelo Base:** [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- **Arquitectura:** DistilBERT es una versión más ligera y rápida de BERT, manteniendo un alto rendimiento en tareas de NLP.
