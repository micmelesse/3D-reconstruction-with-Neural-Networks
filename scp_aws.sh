source config/aws.params
scp -ri $KEY $USER@$DNS:thesis/out/model* ./aws/
ssh -ti $KEY $USER@$DNS "cd thesis; sh clean_models.sh"
