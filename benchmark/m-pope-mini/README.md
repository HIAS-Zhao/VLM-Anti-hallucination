# M-POPE-mini

This folder contains a lightweight benchmark subset derived from the larger local M-POPE experiments.

Included dimensions:

- existence
- color
- number
- position

Each JSON maps an image filename to a list of yes/no questions:

```json
{
  "airport_5.png": {
    "questions": [
      {
        "id": 1,
        "question": "Is there an airport runway in the image?",
        "ground_truth": "yes",
        "sampling_type": "positive"
      }
    ]
  }
}
```

No raw images are included in this repository. Use these files for smoke tests, format validation, and debugging.
