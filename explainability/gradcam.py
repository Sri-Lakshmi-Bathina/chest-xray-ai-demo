import torch
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, class_idx):
        self.model.zero_grad()

        # Forward pass
        output = self.model(self.input_tensor)

        # Backward pass for target class
        output[:, class_idx].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]

        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam

    def set_input(self, input_tensor):
        self.input_tensor = input_tensor


def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay
