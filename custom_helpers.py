import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDatasetLoader(Dataset):
    def __init__(self, folder_path, image_type='png', transform=None):
        """
        Initialize the ImageDatasetLoader class.

        Args:
            folder_path (str): The path to the folder containing the images.
            image_type (str, optional): The type of images to load. Defaults to 'png'.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        self.folder_path = folder_path
        self.image_type = image_type
        self.transform = transform
        self.images = self.load_images_from_folder()

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get the image at the specified index.

        Args:
            idx (int): The index of the image.

        Returns:
            PIL.Image.Image: The image at the specified index.
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def load_images_from_folder(self):
        """
        Load all images from the specified folder.

        Returns:
            list: A list of image file paths.
        
        Raises:
            ValueError: If no image files are found in the specified folder.
        """
        images = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith("." + self.image_type):
                images.append(os.path.join(self.folder_path, filename))

        if not images:
            raise ValueError(f"No {self.image_type.upper()} files found in the specified folder.")

        return images
