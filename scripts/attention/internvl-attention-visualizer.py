import sys 
sys.path.append("../")
import torch 
import typer
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from utils.conversation import get_conv_template 
from utils.utils import load_image, generate_box_indexes, generate_attention_plot, generate_image_patch_attentions
import matplotlib
import matplotlib.font_manager
from typing_extensions import Annotated

# to render the chinese system prompt
matplotlib.rcParams['font.family'] = ['WenQuanYi Zen Hei']
plt.rc('axes', labelsize=20)
plt.rc('axes', titlesize=24)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)


IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(
    question: Annotated[str, typer.Argument(help="Generation prompt.")],
    image: Annotated[str, typer.Argument(help="Path to image file.")], 
    outdir: Annotated[str, typer.Argument(help="Output folder for attention maps.")] = "outputs", 
    max_new_tokens: Annotated[int, typer.Argument(help="Maximum number of tokens to generate.")] = 1024, 
    model_size: Annotated[int, typer.Argument(help="Model size, chooses between [1, 2, 4, 8, 26, 40]")] = 2
):
  assert model_size in [1, 2, 4, 8, 26, 40, 76] 
  if model_size == 76:
    model_id = f"OpenGVLab/InternVL2-Llama3-{model_size}B"
  else: 
    model_id = f"OpenGVLab/InternVL2-{model_size}B"

  model = AutoModel.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    # low_cpu_mem_usage=True, 
    # out attentions just works without flash attention 
    use_flash_attn=False, 
    trust_remote_code=True,
  )
  model = model.to(device)
  model.eval()

  tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    use_fast=False,
  )

  output_dir = Path(outdir) / Path(Path(image).name.split(".")[0] + f"_{question}") 
  if output_dir.exists():
    shutil.rmtree(output_dir)
  output_dir.mkdir(parents=True)


  pixel_values, image_crop_boxes = load_image(
    image, 
    max_num=12, 
    save_patches=True, 
    output_dir=output_dir,
  )

  pixel_values = pixel_values.to(torch.bfloat16)
  pixel_values = pixel_values.to(device)
  generation_config = dict(
    max_new_tokens=max_new_tokens, 
    do_sample=False,
    output_attentions=True, 
    output_logits=True,
    return_dict_in_generate=True,
  )

  history = None 
  num_patches_list = None

  if history is None and pixel_values is not None and '<image>' not in question:
    question = "<image>\n" + question

  if num_patches_list is None: 
    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
  assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

  img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
  model.img_context_token_id = img_context_token_id

  template = get_conv_template(model.template)
  template.system_message = model.system_message 
  eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

  history = [] if history is None else history
  for (old_question, old_answer) in history: 
    template.append_message(template.roles[0], old_question)
    template.append_message(template.roles[1], old_answer)
  template.append_message(template.roles[0], question)
  template.append_message(template.roles[1], None)

  query = template.get_prompt()

  for num_patches in num_patches_list: 
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, 1)  

  model_inputs = tokenizer(query, return_tensors='pt')
  input_ids = model_inputs['input_ids'].to(device)
  attention_mask = model_inputs['attention_mask'].to(device)
  generation_config['eos_token_id'] = eos_token_id

  generation_output = model.generate(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    **generation_config
  )

  if generation_config['output_attentions'] or generation_config['output_logits']: 
    response = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)[0]
  else: 
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

  response = response.split(template.sep)[0].strip()
  history.append((question, response))
  print(f"User: {question}\nAssistant: {response}")


  aggregated_attentions = []
  for attention in generation_output.attentions: 
    averaged_attn = []
    for layer in attention: 
      layer_attns = layer.squeeze(0)
      attns_per_head = layer_attns.mean(dim=0)
      vector = torch.concat((
        # zero the first entry: null attention 
        torch.tensor([0.]).to(device), 
        # usually there is just one item in attns_per_head
        # but on the first generation there is a row for each token 
        # in the prompt as well, so take -1
        attns_per_head[-1][1:],
        # add zero for the final generated token, which never gets any attention 
        torch.tensor([0.]).to(device)
      )) 
      averaged_attn.append(vector / vector.sum())

    aggregated_attention = torch.stack(averaged_attn).mean(dim=0)
    aggregated_attentions.append(aggregated_attention)

  # now pad and stack the attentions 
  max_len = max([v.shape[0] for v in aggregated_attentions])

  padded_without_prompt = torch.stack(
    [
      torch.concat(
        (
          vec, torch.zeros(max_len - vec.shape[0]).to(device)
        )
      ) for vec in aggregated_attentions
    ]
  )

  response_ids = tokenizer(response)

  start_img_token = (model_inputs['input_ids'][0] == 92544).nonzero(as_tuple=True)[0].item()
  end_img_token = (model_inputs['input_ids'][0] == 92545).nonzero(as_tuple=True)[0].item()
  
  system_prompt_indexes = (0, start_img_token - 1)
  query_indexes = (end_img_token + 1, model_inputs['input_ids'][0].shape[0])
  box_indexes = generate_box_indexes(
    pixel_values=pixel_values, 
    start=start_img_token,
  )
  
  generate_attention_plot(
    tokenizer=tokenizer, 
    attention=padded_without_prompt,
    indexes=system_prompt_indexes,
    response_ids=response_ids,
    model_inputs=model_inputs,
    title="Attention Map - System Prompt",
    xlabel="System Prompt Tokens",
    ylabel="Response Tokens",
    out_path=output_dir / Path("attention_map_system_prompt.png")
  ) 

  generate_attention_plot(
    tokenizer=tokenizer,
    attention=padded_without_prompt,
    indexes=query_indexes,
    response_ids=response_ids,
    model_inputs=model_inputs,
    title="Attention Map - Query",
    xlabel="Query Tokens",
    ylabel="Response Tokens",
    out_path=output_dir / Path("attention_map_query.png")
  )

  generate_image_patch_attentions(
    attention=padded_without_prompt,
    indexes=box_indexes, 
    image_crops=image_crop_boxes, 
    output_dir=output_dir, 
    image_size=448,
  )


if __name__ == "__main__": 
  typer.run(main)