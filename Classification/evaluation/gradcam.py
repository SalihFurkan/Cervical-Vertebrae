import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor, mode='classify')
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-10)
        return heatmap.cpu().detach().numpy()

def overlay_heatmap(heatmap, image, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_rgb = np.array(image)
    if len(image_rgb.shape) == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    superimposed_img = heatmap * alpha + image_rgb
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def classify_uploaded_image(model, image_path):
    from data.preprocessor import XRayPreprocessor
    from PIL import Image
    import matplotlib.pyplot as plt

    preprocessor = XRayPreprocessor(train=False, target_size=256)
    image = Image.open(image_path).convert('L')
    input_tensor = preprocessor(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, model.cbam)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor, mode='classify')
        _, predicted = torch.max(outputs, 1)
        class_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']
        predicted_class = class_names[predicted.item()]
        probabilities = outputs.squeeze().cpu().numpy()
        confidence = probabilities[predicted.item()] * 100

    heatmap = grad_cam.generate_heatmap(input_tensor, predicted.item())
    image_np = np.array(image)
    superimposed_img = overlay_heatmap(heatmap, image_np)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f'Grad-CAM: {predicted_class} ({confidence:.2f}%)')
    plt.axis('off')
    plt.show()

    print(f"Predicted Class: {predicted_class} with {confidence:.2f}% confidence")
    return predicted_class, confidence, probabilities

def process_test_dataset_with_gradcam(model, test_dataset, train_dataset_full, device, output_dir="GradCAM/test"):
    os.makedirs(output_dir, exist_ok=True)
    grad_cam = GradCAM(model, model.cbam)

    for idx in range(len(test_dataset)):
        image, true_label = test_dataset[idx]
        true_label = true_label if isinstance(true_label, int) else true_label.item()
        input_tensor = image.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor, mode='classify')
            _, predicted = torch.max(outputs, 1)
            class_names = ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']
            predicted_class = class_names[predicted.item()]
            true_class = class_names[true_label]
            probs = F.softmax(outputs)
            probabilities = probs.squeeze().cpu().numpy()
            confidence = probabilities[predicted.item()] * 100

        heatmap = grad_cam.generate_heatmap(input_tensor, true_label)
        image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, 0]
        superimposed_img = overlay_heatmap(heatmap, image_np)

        filename = f"test_image_{idx}_true_{true_class}_pred_{predicted_class}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
#        print(f"Saved {output_path} - True: {true_class}, Predicted: {predicted_class} ({confidence:.2f}%)")

    print("GradCam Saved")
