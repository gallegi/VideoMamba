import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# print(model)
# load video
video_path = "/teamspace/studios/this_studio/PracticalML_2024/data/raw/UCF101_subset/val/Archery/v_Archery_g18_c02.avi"
container = av.open(video_path)

# extract evenly spaced frames from video
seg_len = container.streams.video[0].frames
clip_len = model.config.encoder.num_frames
indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    if i in indices:
        frames.append(frame.to_ndarray(format="rgb24"))

print(len(frames))

# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 50, 
    "num_beams": 8,
}
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
encoder_out = model.encoder(pixel_values)
print('encoder_out shape: ', encoder_out.last_hidden_state.shape)
print(pixel_values.shape)
tokens = model.generate(pixel_values, **gen_kwargs)
print(len(tokens[0]))
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(caption) 