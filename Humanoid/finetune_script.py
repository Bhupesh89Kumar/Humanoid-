import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Load the Dataset
with open('/Users/bhupeshkumar/Humanoid/dsa_dataset.json', 'r') as file:
    dataset = json.load(file)

# Step 2: Load a Pre-trained Model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Step 3: Tokenize the Dataset
tokenized_dataset = tokenizer([item['question'] for item in dataset], return_tensors='pt', padding=True, truncation=True)

# Step 4: Fine-Tune the Model
# Convert JSON dataset to a text file
text_file_path = '/Users/bhupeshkumar/Humanoid/dsa_dataset.txt'
with open(text_file_path, 'w') as text_file:
    for item in dataset:
        text_file.write(item['question'] + '\n')

# Initialize TextDataset with the text file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=text_file_path,
    block_size=128,
    overwrite_cache=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir='/Users/bhupeshkumar/Humanoid/finetuned_model',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Step 7: Train the Model
trainer.train()

# Step 8: Save the Fine-Tuned Model
model.save_pretrained('/Users/bhupeshkumar/Humanoid/finetuned_model')
tokenizer.save_pretrained('/Users/bhupeshkumar/Humanoid/finetuned_model')
