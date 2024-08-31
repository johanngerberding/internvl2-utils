import numpy as np 
import torch 
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
  transform = T.Compose(
    [
      T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
      T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
      T.ToTensor(),
      T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
  )
  return transform


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple], width: int, height: int, image_size: int) -> tuple:
  best_ratio_diff = float('inf') 
  best_ratio = (1, 1)
  area = width * height 
  for ratio in target_ratios: 
    target_aspect_ratio = ratio[0] / ratio[1]
    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
    if ratio_diff < best_ratio_diff:
      best_ratio_diff = ratio_diff
      best_ratio = ratio 
    elif ratio_diff == best_ratio_diff:
      if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
        best_ratio = ratio 
  
  return best_ratio

def dynamic_preprocess(
  image: Image, 
  min_num: int = 1, 
  max_num: int = 12, 
  image_size: int = 448, 
  use_thumbnail: bool = False,
) -> list:
  org_width, org_height = image.size 
  aspect_ratio = org_width / org_height

  # calculate the existing image aspect ratio 
  target_ratios = set(
    (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
  )
  target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

  # find the closest aspect ratio to the target 
  target_aspect_ratio = find_closest_aspect_ratio(
    aspect_ratio=aspect_ratio, 
    target_ratios=target_ratios, 
    width=org_width, 
    height=org_height, 
    image_size=image_size,
  )

  # calculate the target input width and height 
  target_width = image_size * target_aspect_ratio[0]
  target_height = image_size * target_aspect_ratio[1]
  blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

  resized_img = image.resize((target_width, target_height))
  processed_images = []

  for i in range(blocks): 
    box = (
      (i % (target_width // image_size)) * image_size,
      (i // (target_width // image_size)) * image_size,
      ((i % (target_width // image_size)) + 1) * image_size,
      ((i // (target_width // image_size)) + 1) * image_size,
    )

    # split the image
    split_img = resized_img.crop(box)
    processed_images.append(split_img)

  assert len(processed_images) == blocks 
  if use_thumbnail and len(processed_images) != 1:
    thumbnail_img = image.resize((image_size, image_size))
    processed_images.append(thumbnail_img)
  
  return processed_images


def load_image(image_file: str | Path, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
  """Splits the image based on the aspect ratio into multiple images and transforms it into a tensor.""" 
  image = Image.open(image_file).convert('RGB')
  transform = build_transform(input_size=input_size)
  images = dynamic_preprocess(image=image, image_size=input_size, use_thumbnail=True, max_num=max_num)
  pixel_values = [transform(image) for image in images] 
  pixel_values = torch.stack(pixel_values) 
  return pixel_values

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "OpenGVLab/InternVL2-2B"
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


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

test_image_path = "assets/rechnung_001.png"
pixel_values = load_image(test_image_path, max_num=12).to(torch.bfloat16)
pixel_values = pixel_values.to(device)
generation_config = dict(
  max_new_tokens=1024, 
  do_sample=False,
  output_attentions=True, 
  output_logits=True,
  return_dict_in_generate=True,
)

question = "<image>\nBitte gib mir die Rechnungsnummer." 
response, history, generation_output, model_inputs = model.chat(
  tokenizer, pixel_values, question, generation_config, history=None, return_history=True, verbose=True
)
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

# add prompt attentions
prompt_attention = [
  torch.tensor(
    [1 if i == j else 0 for j, _ in enumerate(model_inputs['input_ids'][0])]
    ).to(device) 
  for i, _ in enumerate(model_inputs['input_ids'][0])
]

attentions = prompt_attention + aggregated_attentions

padded_attentions = torch.stack(
  [
    torch.concat((vec, torch.zeros(max_len - vec.shape[0]).to(device))) 
    for vec in attentions
  ]
)

padded_without_prompt = torch.stack(
  [
    torch.concat(
      (
        vec, torch.zeros(max_len - vec.shape[0]).to(device)
      )
    ) for vec in aggregated_attentions
  ]
)

print(f"Padded without prompt: {padded_without_prompt.shape}")

sparse = padded_attentions.to_sparse()
print(f"Sparse shape: {sparse.shape}")

response_ids = tokenizer(response)
print(f"Response tokens: {response_ids['input_ids']}")
print(f"Response decoded: {[tokenizer.decode(i) for i in response_ids['input_ids']]}")
fig, ax = plt.subplots(figsize=(36, 12))
heatmap = ax.pcolor(padded_without_prompt.cpu().numpy(), cmap=plt.cm.Blues, alpha=0.9)
yticks = [tokenizer.decode(i) for i in response_ids['input_ids']]
ax.set_yticks(range(0, len(yticks)), minor=False)
ax.set_yticklabels(yticks) 
plt.title("Attention heatmap")
fig.savefig("attention.png")
plt.close(fig)

