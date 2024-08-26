from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForLanguageModeling

# from transformers import LlamaTokenizer, LlamaForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from adapters import AdapterConfig




from config import config_dict

def main():
    # Load the saved dataset
    dataset_dict = load_from_disk(config_dict['prepare_dataset']['dataset_save_path'])

    # train, validation, and test sets
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['validation']
    test_dataset = dataset_dict['test']

    # Load pre-trained model and tokenizer
    model_name = 'gpt2'  # model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # # Load the tokenizer and model
    # model_name = "meta-llama/Llama-2-7b"  # Replace with the exact model path
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = LlamaForConditionalGeneration.from_pretrained(model_name)

    # Configure QLoRA
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=16, 
        lora_dropout=0.1,  
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Configure Adapter
    adapter_config = AdapterConfig(
        input_dim=model.config.hidden_size,
        output_dim=model.config.hidden_size,
        adapter_dim=512,  # Dimension of the adapter layer
        activation="relu"  # Activation function for the adapter
    )
    model.add_adapter("my_adapter", adapter_config)


    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        evaluation_strategy='steps',
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Evaluate on the test set
    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    # Save the model
    tokenizer.save_pretrained('./fine-tuned-model')
    model.save_adapter('./my_adapter', "my_adapter")
    model.save_pretrained('./fine-tuned-model-lora')


if __name__ == "__main__":

    main()


