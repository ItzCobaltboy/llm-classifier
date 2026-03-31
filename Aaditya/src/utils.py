from datasets import load_dataset

def load_data():
    dataset = load_dataset("parquet", data_files={
        "train": "Dataset/Dataset_small/train/train.parquet",
        "validation": "Dataset/Dataset_small/val/val.parquet",
        "test": "Dataset/Dataset_small/test/test.parquet"
    })
    return dataset


def encode_labels(dataset):
    labels = list(set(dataset["train"]["generated_by"]))  # FIXED COLUMN
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    def encode(example):
        example["labels"] = label2id[example["generated_by"]]  # FIXED
        return example

    dataset = dataset.map(encode)
    return dataset, label2id, id2label


def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        return tokenizer(
            example["text"],   # FIXED COLUMN
            truncation=True,
            padding="max_length",
            max_length=256
        )

    return dataset.map(tokenize, batched=True)