dns=$(cat dns.config)
ssh -i "thesis.pem" ec2-user@$dns