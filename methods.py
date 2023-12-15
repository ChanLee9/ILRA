import loralib as lora
import modified_modules.krona_layer as krona

def apply_lora(base_model, config):
    if config.task_type == "NLU":
        num_layers = len(base_model.encoder.layer)
        for ly in range(num_layers):
            # query
            query_dim = base_model.encoder.layer[ly].attention.self.query.weight.shape
            base_model.encoder.layer[ly].attention.self.query = lora.Linear(
                in_features=query_dim[0],
                out_features=query_dim[1],
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.dropout
            )
            # value
            value_dim = base_model.encoder.layer[ly].attention.self.value.weight.shape
            base_model.encoder.layer[ly].attention.self.value = lora.Linear(
                in_features=value_dim[0],
                out_features=value_dim[1],
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.dropout
            )
    # mark only lora module as trainable
    lora.mark_only_lora_as_trainable(base_model)


def apply_krona(base_model, config):
    if config.task_type == "NLU":
        num_layers = len(base_model.encoder.layer)
        for ly in range(num_layers):
            # query
            query_dim = base_model.encoder.layer[ly].attention.self.query.weight.shape
            base_model.encoder.layer[ly].attention.self.query = krona.Linear(
                in_features=query_dim[0],
                out_features=query_dim[1],
                krona_dim=config.krona_dim,
                krona_alpha=config.krona_alpha,
                krona_dropout=config.krona_dropout
            )
            # value
            value_dim = base_model.encoder.layer[ly].attention.self.value.weight.shape
            base_model.encoder.layer[ly].attention.self.value = krona.Linear(
                in_features=value_dim[0],
                out_features=value_dim[1],
                krona_dim=config.krona_dim,
                krona_alpha=config.krona_alpha,
                krona_dropout=config.krona_dropout
            )
    # mark only krona module as trainable
    krona.mark_only_krona_as_trainable(base_model)

def apply_pa(base_model, config):
    num_layers = len(base_model.encoder.layer)
    for ly in range(num_layers):
        # query
        query_dim = base_model.encoder.layer[ly].attention.self.query.weight.shape
        base_model.encoder.layer[ly].attention.self.query = lora.Linear(
            in_features=query_dim[0],
            out_features=query_dim[1],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
        # value
        value_dim = base_model.encoder.layer[ly].attention.self.value.weight.shape
        base_model.encoder.layer[ly].attention.self.value = lora.Linear(
            in_features=value_dim[0],
            out_features=value_dim[1],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
        # key
        key_dim = base_model.encoder.layer[ly].attention.self.key.weight.shape
        base_model.encoder.layer[ly].attention.self.key = lora.Linear(
            in_features=key_dim[0],
            out_features=key_dim[1],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
        # output
        attention_output_dim = base_model.encoder.layer[ly].attention.output.dense.weight.shape
        base_model.encoder.layer[ly].attention.output.dense = lora.Linear(
            in_features=attention_output_dim[0],
            out_features=attention_output_dim[1],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
        # ffn1
        ffn1_dim = base_model.encoder.layer[ly].intermediate.dense.weight.shape
        # (3072, 768)
        base_model.encoder.layer[ly].intermediate.dense = lora.Linear(
            in_features=ffn1_dim[1],
            out_features=ffn1_dim[0],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
        # ffn2
        ffn2_dim = base_model.encoder.layer[ly].output.dense.weight.shape
        # (768, 3072)
        base_model.encoder.layer[ly].output.dense = lora.Linear(
            in_features=ffn2_dim[1],
            out_features=ffn2_dim[0],
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.dropout
        )
    
    # mark only lora module as trainable
    lora.mark_only_lora_as_trainable(base_model)


def apply_bit_fit(base_model, config):
    # freeze all parameters except bias
    if config.task_type == "NLU":
        # freeze all parameters
        for _, p in base_model.named_parameters():
            p.requires_grad = False
        
        # unfreeze bias
        num_layers = len(base_model.encoder.layer)
        for ly in range(num_layers):
            base_model.encoder.layer[ly].attention.self.query.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.self.value.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.self.key.bias.requires_grad = True
            base_model.encoder.layer[ly].attention.output.dense.bias.requires_grad = True
            base_model.encoder.layer[ly].intermediate.dense.bias.requires_grad = True
            base_model.encoder.layer[ly].output.dense.bias.requires_grad = True
        base_model.pooler.dense.bias.requires_grad = True
