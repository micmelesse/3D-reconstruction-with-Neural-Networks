dns=$(cat aws.config)
ssh -i "thesis.pem" ec2-user@$dns