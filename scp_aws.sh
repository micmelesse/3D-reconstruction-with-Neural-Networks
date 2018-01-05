rm -rf ./train_dir_aws/*
rm -rf ./out_dir_aws/*
dns=$(cat aws.config)
scp -ri thesis.pem ec2-user@$dns:thesis/train_dir/* ./train_dir_aws
scp -ri thesis.pem ec2-user@$dns:thesis/out_dir/* ./out_dir_aws