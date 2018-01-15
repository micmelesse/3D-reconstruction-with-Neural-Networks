source aws.params
scp -ri $key ec2-user@$dns:thesis/logs/* ./aws_logs
scp -ri $key ec2-user@$dns:thesis/train_dir/* ./aws_train
scp -ri $key ec2-user@$dns:thesis/out/* ./aws_out