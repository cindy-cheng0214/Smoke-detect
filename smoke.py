import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='/home/yuting/Smoke100k/validation/smoke100k_H', help='Path to smoke data')
args = parser.parse_args()

# Load the dataset
test_dataset = load_dataset(args.dataset,
                            split='validation')

print(test_dataset)
print(set(test_dataset['label']))

# generate sentences
# labels = test_dataset.info.features['label'].names
# clip_labels = [f"a photo of a {label}" for label in labels]
clip_labels = ["a photo of a smoke free", "a photo of a smoke"]
print(clip_labels)


# initialization
model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# if you have CUDA set it to the active device like this
device = "cuda" if torch.cuda.is_available() else "cpu"
# move the model to the device
model.to(device)

# create label tokens
label_tokens = processor(
    text=clip_labels,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

# encode tokens to sentence embeddings
label_emb = model.get_text_features(**label_tokens)
# detach from pytorch gradient computation
label_emb = label_emb.detach().cpu().numpy()

# normalization
label_emb = label_emb / np.linalg.norm(label_emb, axis=0)

preds = []
batch_size = 32

# calculate inference runtime
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 1+int(len(test_dataset)/batch_size)
timings=[]

for i in tqdm(range(0, len(test_dataset), batch_size)):
    i_end = min(i + batch_size, len(test_dataset))
    images = processor(
        text=None,
        images=test_dataset[i:i_end]['image'],
        return_tensors='pt'
    )['pixel_values'].to(device)

    starter.record()
    img_emb = model.get_image_features(images)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    timings.append(curr_time)

    img_emb = img_emb.detach().cpu().numpy()
    scores = np.dot(img_emb, label_emb.T)
    preds.extend(np.argmax(scores, axis=1))

true_preds = []
for i, label in enumerate(test_dataset['label']):
    if label == preds[i]:
        true_preds.append(1)
    else:
        true_preds.append(0)

mean_syn = sum(timings) / repetitions
print("average inference runtime: ", mean_syn)

print("Accuracy: {}" .format(sum(true_preds) / len(true_preds)))