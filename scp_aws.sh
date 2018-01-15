source aws.params
# scp -ri $key ec2-user@$dns:thesis/logs/* ./aws_logs
# scp -ri $key ec2-user@$dns:thesis/train_dir/* ./aws_train
# scp -ri $key ec2-user@$dns:thesis/out/* ./aws_out
scp -ri $key ec2-user@$dns:cos429_f17_final_project/model* ./aws_model
