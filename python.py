def transform_and_sample(dataset, processor):
    """
    Transforms a dataset by processing images and prints the keys of a sample.

    Args:
        dataset: The dataset to transform.
        processor: The image processor.

    Returns:
        The transformed dataset.
    """
    def transform(example):
        example["pixel_values"] = processor(image=example["image"], return_tensors="pt")["pixel_values"].squeeze()
        return example

    dataset = dataset.with_transform(transform)
    sample = dataset["train"][0]
    print(sample.keys())
    return dataset

# Example usage (replace with your actual dataset and processor):
# dataset = ...  # Load your dataset here
# processor = ... # Load your processor here.

# transformed_dataset = transform_and_sample(dataset, processor)

# Now you can use the transformed_dataset for further processing or training
# print(transformed_dataset) #if needed.
print("Script executed successfully!")