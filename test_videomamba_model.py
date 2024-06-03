import torch
import numpy as np

from videomamba.video_sm.models.videomamba import videomamba_middle

seed = 4217
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
num_frames = 8
img_size = 224

# To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
mamba_model = videomamba_middle(num_frames=num_frames).cuda()

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# print(model.encoder.config)
# print(model.decoder.config.hidden_size)

encoder_config = model.encoder.config
model.encoder = mamba_model
model.encoder.config = encoder_config
model.encoder.main_input_name = "pixel_values"


# load video
video_path = "assets/file_example_MP4_480_1_5MG.mp4"
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

# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 50, 
    "num_beams": 8,
}
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)
encoder_out = model.encoder(pixel_values)
print('encoder_out shape: ', encoder_out)

tokens = model.generate(pixel_values, **gen_kwargs)
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(caption) 

# model.save_pretrained('checkpoints/videomamba_middle')
