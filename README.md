# Attribute-Guided-Multimodal-IR

A repository containing code for analyzing CLIP's ability to retrieve images based on attribute guided text prompts. The approach is outlined as follows:

> Set up the datasets (MSCOC and PascalVOC)

> Run preliminary retrieval runs based on text prompts (prompt modification optional)

> Modify our dataset (stratified subsets) using image corruptions grounded in semantics like "rain", "fog", etc.

> Repeat retrieval runs using CLIP and generate attention heatmaps for the same.
