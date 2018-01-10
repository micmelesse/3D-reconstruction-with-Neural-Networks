dns=$(cat aws.params)
scp -ri thesis.pem ec2-user@$dns:thesis/logs/* ./aws_logs
scp -ri thesis.pem ec2-user@$dns:thesis/train_dir/* ./aws_train
scp -ri thesis.pem ec2-user@$dns:thesis/out/* ./aws_out