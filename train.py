import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, AdamW, get_scheduler
from datasets import load_dataset  # Ensure dataset is loaded

# Load dataset
dataset = load_dataset("imagefolder", data_dir="D:/Images/Cow_breeds")  # Update path if needed

# Initialize data collator
data_collator = DefaultDataCollator()

# Create DataLoader
train_dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True, collate_fn=data_collator)

print("DataLoader successfully created!")

# Load model (Ensure model is defined and imported properly)
from transformers import AutoModelForImageClassification
model_name = "dima806/animal_151_types_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define learning rate scheduler
num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Check if GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()  # Set model to training mode
    
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model(**batch)  # Forward pass
        loss = outputs.loss  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        lr_scheduler.step()  # Adjust learning rate
        optimizer.zero_grad()  # Reset gradients
    
    print(f"Epoch {epoch+1} completed.")
