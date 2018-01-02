dns=$(cat dns.txt)
scp -ri thesis.pem ec2-user@$dns:thesis/train_dir/* ./train_dir_aws