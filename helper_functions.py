import numpy as np
import os
from PIL import Image
from collections import Counter
import tensorflow as tf


def process_and_save(img_list, mask_list, transform, out_img_dir, out_msk_dir):
    for img_p, msk_p in zip(img_list, mask_list):
        base_name = os.path.basename(img_p)
        mask_name = os.path.basename(msk_p)
        
        image = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(msk_p))
        
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        
        out_img_path = os.path.join(out_img_dir, base_name)
        out_mask_path = os.path.join(out_msk_dir, mask_name)
        
        Image.fromarray(transformed_image).save(out_img_path)
        Image.fromarray(transformed_mask).save(out_mask_path)


def process_and_save_targetting_imbalance(img_list, mask_list, transform, out_img_dir, out_msk_dir):
    for img_p, msk_p in zip(img_list, mask_list):
        base_name = os.path.basename(img_p)
        mask_name = os.path.basename(msk_p)

        image = np.array(Image.open(img_p).convert("RGB"))
        mask = np.array(Image.open(msk_p))

        # detect which classes are present (0,80,160,240)
        classes_present = np.unique(mask // 80)

        # default: 1 augmentation
        reps = 1
        if 1 in classes_present or 3 in classes_present:
            reps = 3   # generate 3 augmented versions

        for r in range(reps):
            transformed = transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            if r == 0:
                out_img_path = os.path.join(out_img_dir, base_name)
                out_mask_path = os.path.join(out_msk_dir, mask_name)
            else:
                name, ext = os.path.splitext(base_name)
                mname, mext = os.path.splitext(mask_name)
                out_img_path = os.path.join(out_img_dir, f"{name}_aug{r}{ext}")
                out_mask_path = os.path.join(out_msk_dir, f"{mname}_aug{r}{mext}")

            Image.fromarray(transformed_image).save(out_img_path)
            Image.fromarray(transformed_mask).save(out_mask_path)


def analyze_class_distribution(mask_dirs, scale_factor=80):
    pixel_counter = Counter()
    masks_with_class = Counter()
    total_masks = 0

    for mdir in mask_dirs:
        for fname in os.listdir(mdir):
            if not fname.endswith('.png'):
                continue

            path = os.path.join(mdir, fname)
            mask = np.array(Image.open(path), dtype=np.int64)  # values: 0,80,160,240

            cls_mask = mask // scale_factor
            cls_mask = np.clip(cls_mask, 0, 3)

            total_masks += 1

            unique_classes = np.unique(cls_mask)
            for c in unique_classes:
                masks_with_class[int(c)] += 1

            flat = cls_mask.reshape(-1)
            counts = np.bincount(flat, minlength=4)
            for c in range(4):
                pixel_counter[c] += int(counts[c])

    print(f"Total masks analyzed: {total_masks}\n")

    print("Masks containing each class:")
    for c in range(4):
        print(f"  Class {c}: {masks_with_class[c]} masks")

    print("\nTotal pixel counts per class:")
    total_pixels = sum(pixel_counter.values())
    for c in range(4):
        count = pixel_counter[c]
        pct = 100.0 * count / total_pixels if total_pixels > 0 else 0.0
        print(f"  Class {c}: {count} pixels ({pct:.4f}%)")



def load_image_and_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)  # Grayscale
    mask = tf.cast(mask, tf.uint8)
    
    # One-hot encode the mask (num_classes=4: 0=background, 1=Cat, 2=Dog, 3=Car)
    mask = tf.one_hot(mask, depth=4, axis=-1)
    mask = tf.squeeze(mask, axis=-2)  # Shape: (256, 256, 4)
    
    return img, mask