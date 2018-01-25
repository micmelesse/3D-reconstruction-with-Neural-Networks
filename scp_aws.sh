source config/aws.params
scp -ri $KEY $USER@$DNS:thesis/out/model_* ./aws/
