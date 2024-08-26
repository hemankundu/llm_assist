def print_model_parameters_and_memory(model, batch_size=1, precision_bits=32, context_length=1024):
    total_params = 0
    total_trainable_params = 0
    
    memory_params = 0
    memory_gradients = 0
    memory_optimizer = 0
    memory_activations = 0

    # Bytes per parameter based on precision
    bytes_per_param = precision_bits / 8
    
    print(f"{'Layer Name':<100} {'# Parameters':<20} {'# Trainable':<15} {'Layer Type':<20}")
    print("="*150)
    
    for name, param in model.named_parameters():
        param_count = param.numel()  # Total number of parameters in this layer
        trainable = param.requires_grad  # Check if the parameter is trainable
        trainable_count = param_count if trainable else 0
        
        layer_type = type(param).__name__
        
        # Memory calculations
        layer_memory = param_count * bytes_per_param
        memory_params += layer_memory
        memory_gradients += layer_memory if trainable else 0
        memory_optimizer += 2 * layer_memory if trainable else 0  # Assuming Adam optimizer

        print(f"{name:<100} {param_count:<20} {trainable_count:<15} {layer_type:<20}")
        
        total_params += param_count
        total_trainable_params += trainable_count
    
    # Estimate memory for activations
    model_hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768
    memory_activations = batch_size * context_length * model_hidden_size * bytes_per_param
    
    print("="*150)
    print(f"Total Parameters: {total_params}")
    print(f"Total Trainable Parameters: {total_trainable_params}")
    
    # Total memory required
    total_memory = memory_params + memory_gradients + memory_optimizer + memory_activations
    
    # Convert to GB
    memory_params_gb = memory_params / (1024 ** 3)
    memory_gradients_gb = memory_gradients / (1024 ** 3)
    memory_optimizer_gb = memory_optimizer / (1024 ** 3)
    memory_activations_gb = memory_activations / (1024 ** 3)
    total_memory_gb = total_memory / (1024 ** 3)
    
    print("\nMemory Requirements (in GB):")
    print(f"Parameters Memory: {memory_params_gb:.4f} GB")
    print(f"Gradients Memory: {memory_gradients_gb:.4f} GB")
    print(f"Optimizer Memory: {memory_optimizer_gb:.4f} GB")
    print(f"Activations Memory: {memory_activations_gb:.4f} GB")
    print(f"Total Estimated Memory: {total_memory_gb:.4f} GB")
