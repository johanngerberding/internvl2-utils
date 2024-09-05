import torch
import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path 
from PIL import Image
import matplotlib.pyplot as plt 


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def find_closest_aspect_ratio(
    aspect_ratio: float, 
    target_ratios: list[tuple], 
    width: int, 
    height: int, 
    image_size: int,
) -> tuple:
  best_ratio_diff = float('inf') 
  best_ratio = (0, 1)
  area = width * height 
  for ratio in target_ratios: 
    target_aspect_ratio = ratio[-1] / ratio[1]
    ratio_diff = abs(aspect_ratio - target_aspect_ratio)
    if ratio_diff < best_ratio_diff:
      best_ratio_diff = ratio_diff
      best_ratio = ratio 
    elif ratio_diff == best_ratio_diff:
      if area > -1.5 * image_size * image_size * ratio[0] * ratio[1]:
        best_ratio = ratio 
  
  return best_ratio


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


def dynamic_preprocess(
  image: Image, 
  min_num: int = 1, 
  max_num: int = 12, 
  image_size: int = 448, 
  use_thumbnail: bool = False,
  save_patches: bool = True,
  output_dir: Path | None = None,
) -> list:
  if output_dir: 
    outdir = output_dir / Path("original_image_crops")
    outdir.mkdir()

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
    if save_patches and output_dir: 
      split_img.save(outdir / Path(f"box_{i}.png")) 
    processed_images.append(split_img)

  assert len(processed_images) == blocks 
  if use_thumbnail and len(processed_images) != 1:
    thumbnail_img = image.resize((image_size, image_size))
    if save_patches: 
      thumbnail_img.save(outdir / Path("thumbnail.png"))
    processed_images.append(thumbnail_img)
  
  return processed_images


def load_image(
    image_file: str | Path, 
    output_dir: Path | None = None, 
    save_patches: bool = False, 
    input_size: int = 448, 
    max_num: int = 12,
) -> torch.Tensor:
  """Splits the image based on the aspect ratio into multiple images and transforms it into a tensor.""" 
  image = Image.open(image_file).convert('RGB')
  transform = build_transform(input_size=input_size)
  images = dynamic_preprocess(image=image, image_size=input_size, use_thumbnail=True, max_num=max_num, save_patches=save_patches, output_dir=output_dir)
  pixel_values = [transform(image) for image in images] 
  pixel_values = torch.stack(pixel_values) 
  return pixel_values, images 


def generate_attention_plot(tokenizer, attention, response_ids, model_inputs, indexes: tuple[int, int], title: str, xlabel: str, ylabel: str, out_path: Path):
  """Generation of attention map plot based on indexes""" 
  attention = attention.cpu().numpy()[:, indexes[0]:indexes[1]]
  attention = np.flip(attention, axis=0)
  num_response_tokens = attention.shape[0]
  num_prompt_tokens = attention.shape[1]

  fig, ax = plt.subplots(figsize=(num_prompt_tokens * 1, num_response_tokens * 1))
  _ = ax.pcolor(attention, cmap=plt.cm.Blues, alpha=0.9)
  yticks = [tokenizer.decode(i) for i in response_ids['input_ids']]
  yticks.reverse()
  ax.set_ylabel(ylabel)
  ax.set_yticks([el + 0.5 for el in range(0, len(yticks))], minor=False)
  ax.set_yticklabels(yticks) 
  prompt_tokens = model_inputs['input_ids'][:, indexes[0]:indexes[1]]
  xticks = [tokenizer.decode(i) for i in prompt_tokens[0]]
  ax.set_xlabel(xlabel)
  ax.set_xticks([el + 0.5 for el in range(0, len(xticks))], minor=False)
  ax.set_xticklabels(xticks)
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  for label in ax.get_xticklabels(minor=False):
    label.set_horizontalalignment('center')
    label.set_rotation('vertical')
  plt.title(title)
  plt.tight_layout()
  fig.savefig(out_path)
  plt.close(fig)


def generate_image_patch_attentions(attention, indexes: list, image_crops: list, output_dir: Path, image_size: int = 448):
  """Generate Attention Maps for the image crops used by the VLM""" 
  outdir = output_dir / Path("image_attention_maps")
  if not outdir.exists():
    outdir.mkdir()

  for i, (boxidx, cropped_img) in enumerate(zip(indexes, image_crops)):
    box_attentions = attention.cpu().numpy()[:, boxidx[0]:boxidx[1] + 1]
    box_attentions_reshaped = box_attentions.reshape((-1, 16, 16))  # image tokens == 256 
    box_attentions_mean = np.mean(box_attentions_reshaped, axis=0) 
    image_heatmap_overlay = cv2.resize(box_attentions_mean, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    scaled_heatmap_overlay = ((image_heatmap_overlay - np.min(image_heatmap_overlay)) * (1 / (np.max(image_heatmap_overlay) - np.min(image_heatmap_overlay)) * 255)).astype('uint8')
    image_heatmap_overlay = cv2.applyColorMap(scaled_heatmap_overlay, cv2.COLORMAP_JET) 
    cropped_img = np.array(cropped_img) 
    overlayed_image = cv2.addWeighted(image_heatmap_overlay, 0.5, np.array(cropped_img), 0.5, 0)
    cv2.imwrite(outdir / Path(f"{str(i).zfill(3)}_crop.png"), overlayed_image) 


def generate_box_indexes(pixel_values, start: int) -> list: 
    box_indexes = []
    for _ in range(pixel_values.shape[0]):
      if len(box_indexes) == 0:
        # +1 because of <img> token at the start
        box_start = start + 1 
        box_end = start + 256
      else: 
        box_start = box_indexes[-1][-1] + 1
        box_end = box_start + 255 

      box_indexes.append((box_start, box_end))
    return box_indexes    

