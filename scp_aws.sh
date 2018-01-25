source config/aws.params
scp -ri $KEY $USER@$DNS:thesis/model_* ./aws_models
