source aws.params
scp -ri $key $user@$dns:thesis/model_* ./aws_models
