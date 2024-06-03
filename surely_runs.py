import torch
import numpy as np

from videomamba.video_sm.models.videomamba import videomamba_middle
from videomamba.video_sm.models.videomamba import videomamba_middle, videomamba_small, load_state_dict


import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


seed = 4217
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
num_frames = 16
img_size = 224

# To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
mamba_model = videomamba_small(num_frames=num_frames).cuda()

state_dict = torch.load("checkpoints/videomamba_s16_k400_f16_res224.pth", map_location='cpu')
load_state_dict(mamba_model, state_dict, center=True)
print('Load checkpoint successfully')
device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# load video
video_path = "assets/v_Archery_g12_c03.avi"
container = av.open(video_path)

# extract evenly spaced frames from video
seg_len = container.streams.video[0].frames
clip_len = num_frames

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
encoder_out = mamba_model(pixel_values)
print('encoder_out shape: ', encoder_out.shape)

