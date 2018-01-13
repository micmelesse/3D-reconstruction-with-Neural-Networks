dns=$(cat aws.params)
scp -ri "ml_ami_key.pem" ec2-user@$dns:thesis/logs/* ./aws_logs
scp -ri "ml_ami_key.pem" ec2-user@$dns:thesis/train_dir/* ./aws_train
scp -ri "ml_ami_key.pem" ec2-user@$dns:thesis/out/* ./aws_out