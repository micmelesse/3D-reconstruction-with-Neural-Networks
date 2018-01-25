source config/aws.params
scp -ri $KEY $USER@$DNS:thesis/out/model_* ./aws/$(date +%Y-%m-%d_%H:%M:%S)/
