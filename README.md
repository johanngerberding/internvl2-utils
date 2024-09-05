# InternVL2 Utils 

Some useful scripts for InternVL.

## Attention Visualizer 

I think this can be a handy tool for people who want to do a bit of prompt tuning for InternVL. I think I will build some more useful things for that model family. The example folder contains an example of a simple German invoice. The following image is the thumbnail attention map for that invoice. 

![Attention Map of the thumbnail image](example/rechnung_001_Bitte%20gib%20mir%20die%20Rechnungsnummer./image_attention_maps/006_crop.png)


## Dataset Generation 

If you want to finetune an InternVL2 model yourself (if you really have that much compute you are a blessed rich boy!), you could use the scripts provided in the `datasets` folder. These should turn the Huggingface dataset into the right format for finetuning.

For more info on how to finetune InterVL2, check out their [website](https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html).