from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling

# from transformers import LlamaTokenizer, LlamaForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from adapters import AdapterConfig
# import bitsandbytes as bnb

from config import config_dict
import tools


def main():
    # Load data, model, tokenizer ---------------------------------------------------

    # # Load the tokenizer and model
    # model_name = "meta-llama/Llama-2-7b"  # Replace with the exact model path
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = LlamaForConditionalGeneration.from_pretrained(model_name)


    # Load pre-trained model and tokenizer
    model_name = 'gpt2'  # model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset_dict = load_from_disk(config_dict['prepare_dataset']['dataset_save_path'])

    # train, validation, and test sets
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']
    test_dataset = dataset_dict['test']

    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_dataset_tokenized = val_dataset.map(tokenize_function, batched=True)
    test_dataset_tokenized = test_dataset.map(tokenize_function, batched=True)

    print(tools.print_model_parameters_and_memory(model, batch_size=2))


    # LoRA ---------------------------------------------------
    # Set up the LoRA configuration
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=16,  
        lora_dropout=0.1,  
        target_modules=["c_attn", "c_fc", "c_proj"],  # Target GPT-2's attention and MLP layers
        task_type=TaskType.CAUSAL_LM,  # Task type: Causal Language Modeling
    )

    model = get_peft_model(model, lora_config)
    print(tools.print_model_parameters_and_memory(model, batch_size=2))

    # quantizing specific layers to 4-bit
    # # model = model.to('cuda')  # Ensure the model is on GPU if available

    # # Apply 4-bit quantization to the model
    # for name, module in model.named_modules():
    #     if any(target in name for target in ["c_attn", "c_fc", "c_proj"]):
    #         quantized_module = bnb.nn.Int8Params(module.weight, requires_grad=True)
    #         module.weight = quantized_module

    # # Configure Adapter
    # adapter_config = AdapterConfig(
    #     input_dim=model.config.hidden_size,
    #     output_dim=model.config.hidden_size,
    #     adapter_dim=512,  # Dimension of the adapter layer
    #     activation="relu"  # Activation function for the adapter
    # )
    # model.add_adapter("my_adapter", adapter_config)


    # Train args ---------------------------------------------------
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=1,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        eval_strategy='steps',
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        data_collator=data_collator,
    )

    # Train ---------------------------------------------------
    trainer.train()

    # Eval ---------------------------------------------------
    # Evaluate on the test set
    test_results = trainer.evaluate(test_dataset_tokenized)
    print(test_results)
    
    # Save ---------------------------------------------------
    # Save the model
    tokenizer.save_pretrained('./fine-tuned-model-lora')
    # model.save_adapter('./my_adapter', "my_adapter")
    model.save_pretrained('./fine-tuned-model-lora')