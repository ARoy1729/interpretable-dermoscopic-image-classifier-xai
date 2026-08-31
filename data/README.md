# Data placement

This repository does not include the HAM10000 images because the dataset is large.

Place the dataset like this:

```text
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ...
└── HAM10000_images_part_2/
    ├── ISIC_0034302.jpg
    └── ...
```

The preprocessing script discovers images by `image_id`, so the two image folders can contain the original archive layout.
