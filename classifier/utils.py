from torchvision import transforms

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint["state_dict"])

def get_image_transformed(img):
    tranform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([64, 64]),
        transforms.Normalize(mean=[0.3720, 0.3410, 0.4056],
                            std=[0.2808, 0.2760, 0.3118]),
    ])

    img_transformed = tranform(img)

    return img_transformed