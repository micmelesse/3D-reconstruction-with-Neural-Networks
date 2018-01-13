dns=$(cat aws.params)
ssh -i "ml_ami_key.pem" ec2-user@$dns