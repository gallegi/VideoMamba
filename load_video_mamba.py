from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from videomamba.video_sm.models.videomamba import videomamba_middle, videomamba_small, load_state_dict

class VideoMambaClassification(nn.Module):
    def __init__(self):
        super().__init__()
        # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
        self.num_labels = len(class_labels)
        self.mamba = videomamba_small(num_frames=NUM_FRAMES, num_classes= self.num_labels)
        state_dict = torch.load("VideoMamba/checkpoints/videomamba_s16_k400_f16_res224.pth", map_location='cpu')
        load_state_dict(self.mamba, state_dict, center=True)

    def forward(self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,):

        logits = self.mamba(pixel_values)
        # print(logits.shape)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
model = VideoMambaClassification()