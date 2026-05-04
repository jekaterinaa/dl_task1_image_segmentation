import albumentations as A

train_transform = A.Compose([
    A.Resize(256, 256),
    # A.GaussianBlur(blur_limit=(3, 7), p=0.05),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.1)
])

test_transform = A.Compose([
    A.Resize(256, 256)
])
