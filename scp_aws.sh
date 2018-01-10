dns=$(cat aws.params)
scp -ri thesis.pem ec2-user@$dns:thesis/train_dir/* ./train_dir_aws
scp -ri thesis.pem ec2-user@$dns:thesis/out/* ./out_aws