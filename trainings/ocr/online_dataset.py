from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import random
from PIL import ImageDraw, ImageFont, ImageFilter, Image
import string
from ocr_config import D2R_KEYWORDS, FONT_PATH, CANVAS_H, CANVAS_W
from typing import Any, Optional
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


class LabelTransformer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        # 0 is reserved for CTC 'blank', so we offset letters by 1
        self.char_to_int = {char: i + 1 for i, char in enumerate(alphabet)}
        self.int_to_char = {i + 1: char for i, char in enumerate(alphabet)}
        self.int_to_char[0] = ""  # Blank token

    def encode(self, text):
        return [self.char_to_int[char] for char in text]

    def decode(self, tokens):
        # Basic greedy decoding: collapse repeats and remove blanks
        res = []
        for i, token in enumerate(tokens):
            if token != 0 and (i == 0 or token != tokens[i - 1]):
                res.append(self.int_to_char[token])
        return "".join(res)


def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    # Pad labels with 0 (the blank token)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    label_lengths = torch.cat(label_lengths)
    return images, labels_padded, label_lengths


class Generator:
    def __init__(self, context: Any):
        self.context = context

    def get_random_label(self):
        roll = random.random()

        if roll < 0.5:
            # Category 1: Real Game Word
            return random.choice(D2R_KEYWORDS)

        elif roll < 0.8:
            # Category 2: Random String (Forces CNN to work harder)
            length = random.randint(3, 10)
            return "".join(
                random.choices(string.ascii_letters + string.digits, k=length)
            )

        else:
            # Category 3: Permuted Word (Robustness)
            word = list(random.choice(D2R_KEYWORDS))
            if len(word) > 3:
                idx = random.randint(0, len(word) - 1)
                word[idx] = random.choice(string.ascii_letters)
            return "".join(word)

    def generate_fixed_sample_memory(self, text):
        # 1. create canvas from random background
        img = self.context.get_random_background((CANVAS_W, CANVAS_H))
        img = Image.fromarray(img)
        canvas = img.convert("RGBA")

        # 2. Randomize font size slightly (D2R text varies by UI element)
        font_size = random.randint(23, 26)
        font = ImageFont.truetype(FONT_PATH, font_size)

        # 3. Calculate text position
        # We want to center it vertically, but we can jitter it horizontally
        bbox = font.getbbox(text)  # [left, top, right, bottom]
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Jitter: Randomly offset the text so the model doesn't expect it in the exact center
        max_x_offset = max(0, CANVAS_W - text_w)
        start_x = random.randint(5, 5 + max_x_offset) if max_x_offset > 0 else 5
        start_y = (CANVAS_H - text_h) // 2 - bbox[1]  # Center vertically

        # draw text background
        overlay = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        # Jitter box opacity and padding
        box_opacity = random.randint(100, 150)

        # Ensure box doesn't clip out of canvas either
        box_rect = [
            max(0, start_x - 2),
            max(0, start_y - 2),
            min(CANVAS_W, start_x + text_w + 2),
            min(CANVAS_H, start_y + text_h + 2),
        ]

        draw_ov.rectangle(box_rect, fill=(0, 0, 0, box_opacity))
        img = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(img)
        text_color = random.choice(
            [
                (255, 255, 255),  # normal
                (255, 255, 100),  # rate
                (105, 105, 255),  # magic
                (175, 157, 101),  # unique
                (96, 96, 96),  # socketed
                (0, 255, 0),  # Set
                (255, 164, 0),  # Rune
            ]
        )  # White, Gold, Blue

        draw.text(
            (start_x, start_y), text.capitalize(), fill=text_color, font=font
        )  # Main Text

        # 5. Noise & Blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.4)))

        # 6. Save as Grayscale (reduces model size and noise sensitivity)
        # img = img.convert("L")

        return img.convert("RGB")


class RealData:
    def __init__(self, root_dir):
        self.dataset = self.__load__(root_dir)

    def __load__(self, root_dir):
        print("Loading captured real data")
        data = {}
        directory_path = Path(root_dir)
        for image_path in directory_path.rglob("*.png"):
            dir_name = image_path.parent.name.upper()
            if dir_name in data:
                samples = data[dir_name]
                samples.append(image_path)
            else:
                samples = [image_path]
                data[dir_name] = samples

        label = random.choice(list(data.keys()))
        samples = data[label]
        print(f"{label}:{len(samples)}")
        return data

    def get_sample(self):
        label = random.choice(list(self.dataset.keys()))
        samples = self.dataset[label]
        if samples is not None:
            sample_path = random.choice(samples)
            sample_img = Image.open(sample_path)
            return label, sample_img
        return None, None


class OnlineGameOCRDataset(Dataset):
    def __init__(
        self,
        epoch_size,
        transformer,
        generator: Generator,
        real_data_dir: Optional[str] = None,
    ):
        self.epoch_size = epoch_size  # How many images per "epoch"
        self.transformer = transformer
        self.transform = T.Compose(
            [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
                T.RandomErasing(
                    p=0.5,  # 50% chance to apply it to an image
                    scale=(0.02, 0.1),  # Erase between 2% and 10% of the image area
                    ratio=(0.3, 3.3),  # Aspect ratio of the erased rectangle
                    value=0,  # Fill with black (0) or 'random' for noise
                    inplace=False,
                ),
            ]
        )
        self.generator = generator
        if real_data_dir is not None:
            self.realData = RealData(real_data_dir)
        else:
            self.realData = None

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        roll = random.random()
        if roll < 0.6 and self.realData is not None:
            label, pil_img = self.realData.get_sample()
        else:
            # 1. Generate the label and image in RAM
            label = (
                self.generator.get_random_label().upper()
            )  # Your hybrid logic function
            pil_img = self.generator.generate_fixed_sample_memory(label)

        # 2. Convert to Tensor
        img_tensor = self.transform(pil_img)

        # 3. Encode the label
        encoded_label = torch.LongTensor(self.transformer.encode(label))
        label_len = torch.IntTensor([len(encoded_label)])

        return img_tensor, encoded_label, label_len
