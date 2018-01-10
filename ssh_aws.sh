dns=$(cat aws.params)
ssh -i "thesis.pem" ec2-user@$dns