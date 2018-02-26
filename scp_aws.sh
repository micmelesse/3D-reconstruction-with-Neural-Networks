source params/aws.params
scp -ri $KEY $USER@$DNS:thesis/out/model* ./aws/
