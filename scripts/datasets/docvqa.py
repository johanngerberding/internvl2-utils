import os 
import json
import typer 
from typing_extensions import Annotated
from datasets import load_dataset 
from tqdm import tqdm


def main(
    meta: Annotated[str, typer.Argument(help="Absolute ath to your dataset meta file (json). If this does not exist yet, it will be generated.")],
    images: Annotated[str, typer.Argument(help="Absolute path to directory where to save images.")], 
    labels: Annotated[str, typer.Argument(help="Absolute Path for label jsonl file.")],
    cache: Annotated[str, typer.Argument(help="Directory for the huggingface cache.")] = "/data/huggingface",
    multiples: Annotated[bool, typer.Argument(help="Use the multiple answers to generate multiple samples per image.")] = False
):
    assert labels.endswith(".jsonl"), "Label file has to be .jsonl" 
    # this env variable has a different name in older transformer versions I think 
    os.environ["HF_HOME"] = cache

    info = """INFO: The DocVQA used here (lmms-labl/DocVQA) provides only answers for the validation set, so just that set will be converted.
    DocVQA has provided multiple answers per question, by setting 'multiples' to True you can generate a sample 
    per answer which is like data augmentation.""" 
    print(info) 
    docvqa = load_dataset(
        "lmms-lab/DocVQA", "DocVQA", cache_dir=cache
    )
    print(f"Number of validation samples: {len(docvqa['validation'])}")

    annos = [] 
    for i, sample in enumerate(tqdm(docvqa['validation'])): 
        docid = sample['docId'] 
        page_no = sample['ucsf_document_page_no']
        image_name = f"{docid}_{str(page_no).zfill(4)}.png" 
        image = sample['image']
        width, height = image.size 

        if not multiples: 
            # just use first answer
            answers = [sample['answers'][0]]
        else:    
            answers = sample['answers']
        
        for answer in answers: 
            conversations = [
                {"from": "human", "value": f"<image>\n{sample['question']}"},
                {"from": "gpt", "value": f"{answer}"}
            ]   
            annos.append(
                {
                    "id": i,
                    "image": image_name,
                    "width": width,
                    "height": height,
                    "conversations": conversations
                }
            )
        
        image_out = os.path.join(images, image_name)
        if not os.path.exists(image_out):
            image.save(os.path.join(images, image_name))

    # add DocVQA to meta file
    if not os.path.exists(meta): 
        metadata = {} 
    else: 
        with open(meta, "r") as fp: 
            metadata = json.load(fp) 
    
    if "lmms-lab/DocVQA" not in metadata:   
        metadata["lmms-lab/DocVQA"] = {
            "root": images,
            "annotation": labels,
            "data_augment": False,
            "repeat_time": 1, 
            "length": len(annos)
        } 
    else: 
        raise ValueError("lmms-lab/DocVQA already exists.")

    with open(meta, "w") as fp:
        json.dump(metadata, fp)

    with open(labels, 'w') as fp: 
        for sample in annos: 
            sample_str = json.dumps(sample)
            fp.write(f"{sample_str}\n")


if __name__ == "__main__":
    typer.run(main)

